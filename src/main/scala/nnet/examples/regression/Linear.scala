package nnet.examples.regression

import nnet.examples.plotting._
import nnet.functions._
import nnet.{Network, NetworkSpec, _}

import scala.util.Random

object Linear extends App {

  val Epochs = 10
  val PointsCount = 10

  val LR = LearningRate.constant(0.003)

  val x1 = Array.fill(PointsCount)(10 * Random.nextDouble())
  val generated = x1.map(x => x -> (linear(x) + 5 * Random.nextGaussian()))
  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))

  val linearNetwork = Network(NetworkSpec.linear(List(1, 1), LR))
  train(linearNetwork, trainingData)

  val x2 = 0.0 until 10.0 by 0.1
  val actual = x2.map(x => x -> linear(x))
  val linear = x2.map(x => x -> linearNetwork.feedForward(Array(x)).head)

  plot("Linear",
    dots("generated", generated),
    line("actual", actual),
    line("linear", linear))

  def linear(x: Double): Double = 5 * x - 4

  def train(network: Network, data: Seq[Input]): Unit = {
    val losses = for (epoch <- 1 to Epochs) yield {
      SGD(network, PointsCount, data)
      val loss = evaluate(network, data)
      logger.info(f"[$epoch]: loss: $loss%f")
      epoch -> loss
    }
    plot("Linear", line("loss", losses))
  }

}
