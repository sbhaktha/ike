package org.allenai.dictionary

import akka.actor.{ Actor, ActorContext, ActorSystem, Props }
import akka.io.IO
import akka.pattern.ask
import akka.util.Timeout
import com.typesafe.config.{ Config, ConfigFactory }
import org.allenai.common.Logging
import org.allenai.dictionary.patterns.NamedPattern
import org.allenai.dictionary.persistence.Tablestore
import spray.can.Http
import spray.http.{ CacheDirectives, HttpHeaders, StatusCodes }
import spray.httpx.SprayJsonSupport
import spray.routing.{ ExceptionHandler, HttpService }
import spray.util.LoggingContext
import org.allenai.common.Config._

import scala.collection.JavaConverters._
import scala.concurrent.duration.DurationInt
import scala.concurrent.{ Await, Future }
import scala.language.postfixOps
import scala.util.control.NonFatal

import java.util.concurrent.TimeUnit

object DictionaryToolWebapp {
  lazy val config = ConfigFactory.load().getConfig("DictionaryToolWebapp")
  val name = "dictionary-tool"
  def main(args: Array[String]): Unit = {
    implicit val system = ActorSystem("dictionary-tool")
    val port = config.getInt("port")
    val service = system.actorOf(Props[DictionaryToolActor], "webapp-actor")

    {
      implicit val timeout = Timeout(30.seconds)
      IO(Http) ? Http.Bind(service, interface = "0.0.0.0", port = port)
    }
  }

  implicit class FutureWithGet[T](val future: Future[T]) extends AnyVal {
    def get: T = Await.result(future, 0 nanos)
  }
}

class DictionaryToolActor extends Actor with HttpService with SprayJsonSupport with Logging {
  import DictionaryToolWebapp.FutureWithGet
  import JsonSerialization._
  import spray.json._
  import spray.json.DefaultJsonProtocol._
  import context.dispatcher

  logger.debug("Starting DictionaryToolActor") // this is just here to force logger initialization

  val config = ConfigFactory.load
  val searchApps = config.getConfigList("DictionaryToolWebapp.indices").asScala.map { config =>
    config.getString("name") -> Future { SearchApp(config) }
  }.toMap
  val similarPhrasesSearcher =
    new WordVecPhraseSearcher(ConfigFactory.load()[Config]("SimilarPhrasesSearcher"))

  implicit def myExceptionHandler(implicit log: LoggingContext): ExceptionHandler =
    ExceptionHandler {
      case NonFatal(e) =>
        requestUri { uri =>
          log.error(e, e.getMessage)
          complete(StatusCodes.InternalServerError -> e.getMessage)
        }
    }

  val serviceRoute = pathPrefix("api") {
    parameters('corpora.?) { corpora =>
      val searchersFuture = Future.sequence(corpora match {
        case None => searchApps.values
        case Some(searcherKeys) => searcherKeys.split(' ').map(searchApps).toIterable
      })

      path("groupedSearch") {
        post {
          entity(as[SearchRequest]) { req =>
            complete {
              val query = SearchApp.parse(req).get

              val (tables, patterns) = req.userEmail match {
                case Some(userEmail) => (
                  Tablestore.tables(userEmail),
                  Tablestore.namedPatterns(userEmail)
                )
                case None => (Map.empty[String, Table], Map.empty[String, NamedPattern])
              }

              val interpolatedQuery = QueryLanguage.interpolateQuery(
                query,
                tables,
                patterns,
                similarPhrasesSearcher
              ).get
              val resultsFuture = searchersFuture.map { searchers =>
                val parResult = searchers.par.flatMap { searcher =>
                  searcher.search(interpolatedQuery, req.config).get
                }
                parResult.seq
              }
              val groupedFuture = for {
                results <- resultsFuture
              } yield {
                req.target match {
                  case Some(target) => SearchResultGrouper.groupResults(req, tables, results)
                  case None => SearchResultGrouper.identityGroupResults(req, results)
                }
              }
              val qexpr = SearchApp.parse(req).get
              groupedFuture.map { grouped => SearchResponse(qexpr, grouped) }
            }
          }
        }
      } ~
        path("wordInfo") {
          post {
            entity(as[WordInfoRequest]) { req =>
              complete(searchersFuture.map { searchers =>
                val results = searchers.par.map(_.wordInfo(req).get)

                // find the word
                val word = results.head.word
                require(results.forall(_.word == word))

                // combine the pos tags
                def combineCountMaps[T](left: Map[T, Int], right: Map[T, Int]) =
                  left.foldLeft(right) {
                    case (map, newPair) =>
                      map.updated(newPair._1, map.getOrElse(newPair._1, 0) + newPair._2)
                  }
                val posTags = results.map(_.posTags).reduce(combineCountMaps[String])

                WordInfoResponse(word, posTags)
              })
            }
          }
        } ~
        path("suggestQuery") {
          post {
            entity(as[SuggestQueryRequest]) { req =>
              val timeout = config.getConfig("QuerySuggester").
                getDuration("timeoutInSeconds", TimeUnit.SECONDS)
              complete(searchersFuture.map { searchers =>
                SearchApp.suggestQuery(searchers.toSeq, req, similarPhrasesSearcher, timeout)
              })
            }
          }
        }
    } ~ path("similarPhrases") {
      parameters('phrase) { phrase =>
        complete(SimilarPhrasesResponse(similarPhrasesSearcher.getSimilarPhrases(phrase)))
      }
    }
  }

  val tablesRoute = pathPrefix("api" / "tables") {
    pathPrefix(Segment) { userEmail =>
      path(Segment) { tableName =>
        pathEnd {
          get {
            complete {
              Tablestore.tables(userEmail).get(tableName) match {
                case None => StatusCodes.NotFound
                case Some(table) if table.name == tableName => table
                case _ => StatusCodes.BadRequest
              }
            }
          } ~ put {
            entity(as[Table]) { table =>
              complete {
                if (table.name == tableName) {
                  Tablestore.putTable(userEmail, table)
                } else {
                  StatusCodes.BadRequest
                }
              }
            }
          } ~ delete {
            complete {
              Tablestore.deleteTable(userEmail, tableName)
              StatusCodes.OK
            }
          }
        }
      } ~ pathEndOrSingleSlash {
        get {
          complete {
            Tablestore.tables(userEmail).keys.mkString("\n")
          }
        }
      }
    }
  }

  val patternsRoute = pathPrefix("api" / "patterns") {
    pathPrefix(Segment) { userEmail =>
      path(Segment) { patternName =>
        pathEnd {
          get {
            complete {
              Tablestore.namedPatterns(userEmail).get(patternName) match {
                case None => StatusCodes.NotFound
                case Some(pattern) => pattern.pattern
              }
            }
          } ~ put {
            entity(as[String]) { pattern =>
              complete {
                Tablestore.putNamedPattern(userEmail, NamedPattern(patternName, pattern))
              }
            }
          } ~ delete {
            complete {
              Tablestore.deleteNamedPattern(userEmail, patternName)
              StatusCodes.OK
            }
          }
        }
      } ~
        pathEndOrSingleSlash {
          get {
            complete(Tablestore.namedPatterns(userEmail).values)
          }
        }
    }
  }

  val corporaRoute = path("api" / "corpora") {
    pathEnd {
      complete {
        val readySearchApps = searchApps.filter(_._2.isCompleted)
        readySearchApps.map {
          case (corpusName, app) => CorpusDescription(corpusName, app.get.description).toJson
        }
      }
    }
  }

  val mainPageRoute = pathEndOrSingleSlash {
    getFromFile("public/index.html")
  } ~ get {
    unmatchedPath { p => getFromFile("public" + p) }
  }

  def actorRefFactory: ActorContext = context
  val cacheControlMaxAge = HttpHeaders.`Cache-Control`(CacheDirectives.`max-age`(0))
  def receive: Actor.Receive = runRoute(
    mainPageRoute ~
      serviceRoute ~
      tablesRoute ~
      patternsRoute ~
      corporaRoute
  )
}
