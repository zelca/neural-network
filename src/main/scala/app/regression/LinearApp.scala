package app.regression

import com.typesafe.scalalogging.Logger
import neural._
import neural.function._
import org.sameersingh.scalaplot.Implicits._
import org.slf4j.LoggerFactory

import scala.util.Random

object LinearApp extends App {

  val logger = Logger(LoggerFactory.getLogger("linear-app"))

  val LF = Quadratic
  val LR = LearningRate.constant(0.01)

  val x1 = -5.0 until 5.0 by 0.1
  val actual = x1.map(linear)
  val generated = x1.map(linear(_) + Random.nextGaussian())

  val network = Network(NetworkSpec.linear(Left(List(1, 1)), LR))
  // val network = Network(NetworkSpec(Left(List(1, 1)), LR, Sigmoid, linearOutput = true, Quadratic))

  val trainingData = (x1, generated).zipped.map((xi, yi) => (Array(yi), Array(xi)))
  (0 until 10) foreach {
    i =>
      SGD(network, trainingData, 50)
      logger.info("[%s]: loss: %f".format(i, evaluate(network, trainingData, LF)))
  }

  val x2 = -5.0 until 5.0 by 0.83
  val predicted = x2.map(x => network.feedForward(Array(x)).head)

  output(GUI, xyChart(List(x1 -> Y(actual), x1 -> Y(generated), x2 -> Y(predicted))))

  def linear(x: Double): Double = 5 * x - 4

}
