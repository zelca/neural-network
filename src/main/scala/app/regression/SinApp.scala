package app.regression

import com.typesafe.scalalogging.Logger
import neural._
import neural.function._
import org.sameersingh.scalaplot.Implicits._
import org.slf4j.LoggerFactory

import scala.util.Random

object SinApp extends App {

  val logger = Logger(LoggerFactory.getLogger("sin-app"))

  val LF = Quadratic
  val LR = LearningRate.constant(0.1)

  val x1 = 0.0 until 10.0 by 0.05
  val actual = x1.map(math.sin)
  val generated = x1.map(math.sin(_) + 0.1 * Random.nextGaussian())

  val network = Network(NetworkSpec(Left(List(1, 6, 1)), LR, Sigmoid, linearOutput = true, LF))

  val trainingData = (x1, generated).zipped.map((xi, yi) => (Array(yi), Array(xi)))
  for (epoch <- 0 until 500) {
    SGD(network, 200, trainingData)
    logger.info("[%s]: loss: %f".format(epoch, evaluate(network, trainingData)))
  }

  val net = hardcodedNetwork()
  val hardcoded = x1.map(x => net.feedForward(Array(x)).head)

  val x2 = 0.0 until 10.0 by 0.03
  val predicted = x2.map(x => network.feedForward(Array(x)).head)
  output(GUI, xyChart(List(x1 -> Y(actual), x1 -> Y(generated), x1 -> Y(hardcoded), x2 -> Y(predicted))))

  def hardcodedNetwork(): Network = {
    val layers = List(
      List(
        (-1.55, Array(1.44)),
        (-3.5, Array(-0.02)),
        (-8.4, Array(0.93)),
        (-12.81, Array(1.98)),
        (-0.49, Array(1.84)),
        (9.78, Array(-3.14))),
      List((-2.59, Array(-1.39, 3.85, -3.59, 2.92, 2.94, 1.70)))
    )

    Network(NetworkSpec(Right(layers), None, Sigmoid, linearOutput = true, None))
  }

}

