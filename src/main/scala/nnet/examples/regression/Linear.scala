package nnet.examples.regression

import nnet.examples.plotting._
import nnet.functions._
import nnet.{Network, NetworkSpec, _}

import scala.util.Random

object Linear extends App {

  val LR = LearningRate.constant(0.01)

  val x1 = -5.0 until 5.0 by 0.1
  val actual = x1.map(x => x -> linear(x))
  val generated = x1.map(x => x -> (linear(x) + Random.nextGaussian()))

  val network = Network(NetworkSpec.linear(Left(List(1, 1)), LR))
  // val network = Network(NetworkSpec(Left(List(1, 1)), LR, Sigmoid, linearOutput = true, Quadratic))

  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))
  val losses = for (epoch <- 1 to 20) yield {
    SGD(network, 10, trainingData)
    val loss = evaluate(network, trainingData)
    logger.info(f"[$epoch]: loss: $loss%f")
    epoch.toDouble -> loss
  }
  plot("Linear", Series("loss", losses))

  val x2 = -5.0 until 5.0 by 0.83
  val predicted = x2.map(x => x -> network.feedForward(Array(x)).head)

  plot("Linear",
    Series("actual", actual),
    Series("predicted", predicted),
    Series("generated", generated, dots = true))

  def linear(x: Double): Double = 5 * x - 4

}
