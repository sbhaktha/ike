package org.allenai.ike

import org.allenai.common.Logging

import java.io.{ File, PrintWriter }

import org.allenai.common.Config._
import org.allenai.ike.patterns.NamedPattern
import org.allenai.ike.persistence.Tablestore

import com.typesafe.config.{ Config, ConfigFactory }

import scala.collection.JavaConverters._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.io.Source
import scala.util.{ Try, Success, Failure }

import scopt.OptionParser

/** A CLI that can run specified patterns over selected corpus indices
  * and dumps a table corresponding to each specified pattern to the
  * user-specified directory path. Tables will be in `tsv` format, which
  * is the same format as when we download tables via the IKE UI (Tables
  * tab).
  * The indices are specified in the config file (`application.conf`).
  */
object IkeBatchSearch extends App with Logging {
  /** Arguments to run IkeBatchSearch */
  case class IkeBatchSearchOptions(
    // input file containing an IKE pattern name and a corresponding
    // output table name per line in the following format:
    // <patternName>\t<tableName>
    // <patternName> is expected to refer to one of the named patterns the user
    // with specified account (userEmail) has saved via the IKE web tool.
    patternFile: Option[File] = None,
    // account info to get named patterns from. These patterns might contain
    // references to tables saved in the user's account. Currently IKE does
    // not enforce any security around user data (tables and patterns), so
    // we can look up any user's data. Eventually we should add permissions on
    // tables/patterns and required authentication.
    userEmail: String = "",
    // path to output directory with generated tables
    outputTableDir: String = ""
  )

  val optionParser: OptionParser[IkeBatchSearchOptions] = new scopt.OptionParser[IkeBatchSearchOptions]("") {
    opt[File]("pattern-file") valueName ("<file>") required () action { (patternFileVal, options) =>
      options.copy(patternFile = Some(patternFileVal))
    } text ("Input pattern file")
    opt[String]("user-email") valueName ("<string>") required () action { (userEmailVal, options) =>
      options.copy(userEmail = userEmailVal)
    } text ("IKE User account email")
    opt[String]("output-dir") valueName ("<string>") required () action { (outputDirVal, options) =>
      options.copy(outputTableDir = outputDirVal)
    } text ("Directory path for output tables")
    help("help") text ("Prints this help.")
  }

  optionParser.parse(args, IkeBatchSearchOptions()) map { batchSearchOptions =>
    // Pull up the tables and patterns saved for the provided user account.
    val (tables, patterns) =
      (
        Tablestore.tables(batchSearchOptions.userEmail),
        Tablestore.namedPatterns(batchSearchOptions.userEmail)
      )

    val appConfig = ConfigFactory.load

    // Create a Searcher per index so that search can be parallelized.
    val searchApps = appConfig.getConfigList("IkeToolWebapp.indices").asScala.map { config =>
      config.getString("name") -> Future { SearchApp(config) }
    }.toMap
    val searchersFuture = Future.sequence(searchApps.keys.map(searchApps))

    // Create instance of similar phrase searcher necessary to process queries such as
    // `cat ~ 10` which should return the 10 closest word2vec neighbors to `cat`.
    val word2vecPhrasesSearcher =
      new EmbeddingBasedPhrasesSearcher(appConfig[Config]("word2vecPhrasesSearcher"))

    // Create a table expander. This enables processing queries involving word2vec neighbors
    // for a specified table column's contents. For e.g., if we have a MaterialConductivity
    // table with an 'Energy' column that has some seed entries like "light" and "heat" and
    // we would like to find similar words like "thermal energy" and "electricity", the
    // `tableExpander` will compute the word2vec centroid of the specified column's current
    // entries and pull up n closest neighbors. The query that would trigger this will look
    // like: `$MaterialConductivity.Energy ~ 10`.
    val tableExpander = new SimilarPhrasesBasedTableExpander(word2vecPhrasesSearcher)

    /** Helper to group a given collection of ``BlackLabResult``s by extraction and produce
      * `GroupedBlackLabResult`s
      */
    def getGroupedBlackLabResults(
      blacklabResults: Iterable[BlackLabResult],
      query: QExpr,
      tableName: String
    ): Seq[GroupedBlackLabResult] = {
      // The call to `groupResults` requires us to pass a SearchRequest object, which takes a
      // SearchConfig.
      // Setting `limit` to None because in batch mode we want all extractions.
      val searchConfig = new SearchConfig(limit = None, evidenceLimit = Some(10))
      val searchRequest = new SearchRequest(Right(query), Some(tableName),
        Some(batchSearchOptions.userEmail), searchConfig)
      SearchResultGrouper.groupResults(searchRequest, tables, blacklabResults)
    }

    /** Helper that takes a pattern and runs it over the required searchers in parallel.
      * Writes extractions to the specified output table file.
      */
    def processPatternAndCreateTable(
      pattern: NamedPattern, tableName: String, tableDir: String
    ) = {
      SearchApp.parse(Left(pattern.pattern)) match {
        case Success(query) =>
          // Get a simplified (ready-to-execute) version of the query.
          val interpolatedQuery = QueryLanguage.interpolateQuery(
            query,
            tables,
            patterns,
            word2vecPhrasesSearcher,
            tableExpander
          ).get

          // Run the query over all the searchers in parallel.
          val resultsFuture = searchersFuture.map { searchers =>
            val parResult = searchers.par.flatMap { searcher =>
              searcher.search(interpolatedQuery).get
            }
            parResult.seq
          }

          // Construct tables rows from `BlackLabResult`s, and write to output table file.
          resultsFuture onComplete {
            case Success(blacklabResults) =>
              // Transform `BlackLabResult`s into `GroupedBlackLabResult`s which are grouped
              // by extractions, to construct table rows from.
              val groupedBlacklabResults =
                getGroupedBlackLabResults(blacklabResults, query, tableName)

              // Construct table rows from the `GroupedBlackLabResult`s.
              // In the regular IKE workflow via the web tool, a row has to be positive or negative
              // based on user annotation. Without either of these a row will not get added to the
              // table. Creating an 'unlabeled' category for the batch mode scenario. Assumption
              // here is that the extractions may be vetted later.
              // TODO: Save provenance.
              // We are currently not saving the provenance (`provenance` column, which is optional,
              // will be empty). There is no data model defined for provenance.
              // It is simply defined as an optional JsValue, which makes it messy to construct and
              // write out. Also, the size of the table would blow up if we add provenance, so we
              // may not be able to load the created table into IKE for demo purposes.
              val tableRows = groupedBlacklabResults.map(_.keys ++ Seq("unlabeled", ""))

              // Write to file.
              val tableOpFile = new File(tableDir, tableName + ".tsv")
              val tableFileWriter = new PrintWriter(tableOpFile)
              // Get Column Names for table header.
              val columnNames =
                SearchResultGrouper.generateColumnNamesForNewTable(
                  groupedBlacklabResults.head.keys.size
                )
              // Create header row with column names. Add columns corresponding to
              // label and provenance. These are expected as part of the header when we try to load
              // the table into IKE.
              val headerRow = (columnNames ++ Seq("label", "provenance")).mkString("\t")
              tableFileWriter.println(headerRow)
              tableRows.map(row => tableFileWriter.println(row.mkString("\t")))
              tableFileWriter.close()

            case Failure(f) => logger.error(s"Unable to parse pattern  ${pattern.name}")
          }
        case Failure(f) => logger.error(s"Unable to parse pattern  ${pattern.name}")
      }
    }

    // Iterate over the patterns in the pattern file, process each pattern to produce
    // a corresponding output table.
    batchSearchOptions.patternFile map { patternFile =>
      val patternLines = Source.fromFile(patternFile).getLines()
      for (patternLine <- patternLines) {
        patternLine.split("\t").map(_.trim) match {
          case Array(patternName, tableName, _*) =>
            logger.info(s"Processing pattern $patternName to produce table $tableName")
            patterns.get(patternName) match {
              case Some(pattern) =>
                processPatternAndCreateTable(pattern, tableName, batchSearchOptions.outputTableDir)
              case None => logger.error(s"Pattern $patternName not found!")
            }
          case _ => logger.error(s"Invalid format in pattern file, line: $patternLine")
        }
      }
    }
  }
}
