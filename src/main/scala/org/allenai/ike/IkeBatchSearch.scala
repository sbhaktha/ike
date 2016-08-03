package org.allenai.ike

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

object IkeBatchSearch extends App {
  /** Arguments to run IkeBatchSearch */
  case class IkeBatchSearchOptions(
    // input file containing an IKE pattern name and a corresponding
    // output table name per line in the following format:
    // <patternName>\t<tableName>
    // <patternName> is expected to refer to one of the named patterns the user
    // with specified account (userEmail) has saved via the IKE web tool.
    patternFile: Option[File] = None,
    // account info
    userEmail: String = "",
    // path to output directory with generated tables
    outputTableDir: String = ""
  )

  val config = ConfigFactory.load
  val searchApps = config.getConfigList("IkeToolWebapp.indices").asScala.map { config =>
    config.getString("name") -> Future { SearchApp(config) }
  }.toMap

  // Create instances of embedding based similar phrase searchers
  val word2vecPhrasesSearcher =
    new EmbeddingBasedPhrasesSearcher(ConfigFactory.load()[Config]("word2vecPhrasesSearcher"))

  val tableExpander = new SimilarPhrasesBasedTableExpander(word2vecPhrasesSearcher)

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

  optionParser.parse(args, IkeBatchSearchOptions()) match {
    case Some(batchSearchOptions) =>
      val (tables, patterns) =
        (
          Tablestore.tables(batchSearchOptions.userEmail),
          Tablestore.namedPatterns(batchSearchOptions.userEmail)
        )
      val tableNames = tables.map(_._1).mkString(", ")
      val patternNames = patterns.keySet
      println(s"Patterns for user ${batchSearchOptions.userEmail}: $patternNames")
      println(s"Tables for user ${batchSearchOptions.userEmail}: $tableNames")
      val corpusNames = searchApps.keys
      println(s"Corpus names: ${corpusNames.mkString(", ")}")
      val searchersFuture = Future.sequence(corpusNames.map(searchApps))

      println(s"patternFile: ${batchSearchOptions.patternFile}")
      batchSearchOptions.patternFile map { patternFile =>
        val patternLines = Source.fromFile(patternFile).getLines()
        for (patternLine <- patternLines) {
          println(s"pattern line: $patternLine")
          patternLine.split("\t").map(_.trim) match {
            case Array(patternName, tableName, _*) =>
              println(s"Processing pattern $patternName to produce table $tableName")
              patterns.get(patternName) match {
                case Some(pattern) =>
                  SearchApp.parse(Left(pattern.pattern)) match {
                    case Success(query) =>
                      println(s"query: $query")
                      val interpolatedQuery = QueryLanguage.interpolateQuery(
                        query,
                        tables,
                        patterns,
                        word2vecPhrasesSearcher,
                        tableExpander
                      ).get
                      val resultsFuture = searchersFuture.map { searchers =>
                        val parResult = searchers.par.flatMap { searcher =>
                          searcher.search(interpolatedQuery).get
                        }
                        parResult.seq
                      }
                      val searchConfig = new SearchConfig(limit = None, evidenceLimit = Some(10))
                      val searchRequest = new SearchRequest(Right(query), Some(tableName),
                        Some(batchSearchOptions.userEmail), searchConfig)
                      for {
                        results <- resultsFuture
                      } yield {
                        val groupedBlacklabResults: Seq[GroupedBlackLabResult] =
                          SearchResultGrouper.groupResults(searchRequest, tables, results)
                        println(s"groupedBlacklabResults size: ${groupedBlacklabResults.size}")
                        val columnNames =
                          SearchResultGrouper.generateColumnNamesForNewTable(
                            groupedBlacklabResults.head.keys.size
                          )
                        println(s"columnNames: ${columnNames}")
                        val tableRows: Seq[Seq[String]] =
                          groupedBlacklabResults.map(_.keys ++ Seq("unlabeled", ""))
                        println(s"Found ${tableRows.size} rows")
                        val tableFileWriter =
                          new PrintWriter(new File(batchSearchOptions.outputTableDir, tableName + ".tsv"))
                        // Create header row with column names.
                        val headerRow = (columnNames ++ Seq("label", "provenance")).mkString("\t")
                        tableFileWriter.println(headerRow)
                        tableRows.map(row => tableFileWriter.println(row.mkString("\t")))
                        tableFileWriter.close()
                      }
                    case Failure(f) => println(s"Unable to parse pattern  $patternName")
                  }
                case None => println(s"Pattern $patternName not found!")
              }
            case _ => println(s"Invalid format in pattern file, line: $patternLine")
          }
        }
      }
    case None => println(s"Unspecified pattern file")
  }
}
