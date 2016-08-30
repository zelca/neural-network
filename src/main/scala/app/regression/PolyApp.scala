package app.regression

import com.typesafe.scalalogging.Logger
import neural._
import neural.function._
import org.sameersingh.scalaplot.Implicits._
import org.slf4j.LoggerFactory

import scala.util.Random

object PolyApp extends App {

  val logger = Logger(LoggerFactory.getLogger("poly-app"))

  val LF = Quadratic
  val LR = LearningRate.constant(0.01)

  val x1 = -2.8 until 2.8 by 0.05
  val actual = x1.map(polynomial)
  val generated = x1.map(polynomial(_) + Random.nextGaussian())

  val network = Network(NetworkSpec(Left(List(1, 10, 1)), LR, linearOutput = true, lossFunction = LF))

  val trainingData = (x1, generated).zipped.map((xi, yi) => (Array(yi), Array(xi)))
  for (epoch <- 0 until 500) {
    SGD(network, 100, trainingData)
    logger.info("[%s]: loss: %f".format(epoch, evaluate(network, trainingData)))
  }

  val x2 = -2.8 until 2.8 by 0.03
  val predicted = x2.map(x => network.feedForward(Array(x)).head)

  output(GUI, xyChart(List(x1 -> Y(actual), x1 -> Y(generated), x2 -> Y(predicted))))

  def polynomial(x: Double): Double = math.pow(x, 5) - 8 * math.pow(x, 3) + 10 * x + 6

}

