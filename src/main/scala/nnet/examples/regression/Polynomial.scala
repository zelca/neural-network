package nnet.examples.regression

import nnet.examples.plotting._
import nnet.functions._
import nnet.{Network, NetworkSpec, _}

import scala.util.Random

object Polynomial extends App {

  val Epochs = 500
  val PointsCount = 100

  val LF = Quadratic
  val LR = LearningRate.constant(0.01)

  val x1 = Array.fill(PointsCount)(-2.8 + 5.6 * Random.nextDouble())
  val generated = x1.map(x => x -> (polynomial(x) + Random.nextGaussian()))
  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))

  val network = Network(NetworkSpec(List(1, 10, 1), LR, linearOutput = true, lossFunction = LF))
  train(network, trainingData)

  val x2 = -2.8 until 2.8 by 0.05
  val actual = x2.map(x => x -> polynomial(x))
  val predicted = x2.map(x => x -> network.feedForward(Array(x)).head)

  plot("Polynomial",
    dots("generated", generated),
    line("actual", actual),
    line("predicted", predicted))

  def polynomial(x: Double): Double =
    math.pow(x, 5) - 8 * math.pow(x, 3) + 10 * x + 6

  def train(network: Network, data: Seq[Input]): Unit = {
    val losses = for (epoch <- 1 to Epochs) yield {
      SGD(network, PointsCount, data)
      val loss = evaluate(network, data)
      logger.info(f"[$epoch]: loss: $loss%f")
      epoch -> loss
    }
    plot("Polynomial", line("loss", losses))
  }

}

