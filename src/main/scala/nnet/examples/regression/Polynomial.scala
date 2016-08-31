package nnet.examples.regression

import nnet.examples.plotting._
import nnet.functions._
import nnet.{Network, NetworkSpec, _}

import scala.util.Random

object Polynomial extends App {

  val LF = Quadratic
  val LR = LearningRate.constant(0.01)

  val x1 = -2.8 until 2.8 by 0.05
  val actual = x1.map(x => x -> polynomial(x))
  val generated = x1.map(x => x -> (polynomial(x) + Random.nextGaussian()))

  val network = Network(NetworkSpec(Left(List(1, 10, 1)), LR, linearOutput = true, lossFunction = LF))

  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))
  val losses = for (epoch <- 1 to 500) yield {
    SGD(network, 100, trainingData)
    val loss = evaluate(network, trainingData)
    logger.info(f"[$epoch]: loss: $loss%f")
    epoch.toDouble -> loss
  }
  plot("Polynomial", Series("loss", losses))

  val x2 = -2.8 until 2.8 by 0.03
  val predicted = x2.map(x => x -> network.feedForward(Array(x)).head)

  plot("Polynomial",
    Series("actual", actual),
    Series("predicted", predicted),
    Series("generated", generated, dots = true))

  def polynomial(x: Double): Double = math.pow(x, 5) - 8 * math.pow(x, 3) + 10 * x + 6

}

