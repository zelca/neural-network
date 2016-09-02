package nnet.examples.regression

import nnet.examples.plotting._
import nnet.functions._
import nnet.{Network, NetworkSpec, _}

import scala.util.Random

object Linear extends App {

  val LF = Quadratic
  val LR = LearningRate.constant(0.1)
  val L2R = new L2(0.8)

  val x1 = -5.0 until 5.0 by 0.1
  val generated = x1.map(x => x -> (linear(x) + 5 * Random.nextGaussian()))
  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))

  val linearNetwork = Network(NetworkSpec.linear(List(1, 1), LR))
  train(linearNetwork)

  val overfittedNetwork = Network(NetworkSpec(List(1, 3, 1), LR, Sigmoid, linearOutput = true, LF))
  train(overfittedNetwork)

  val fixedNetwork = Network(NetworkSpec(List(1, 3, 1), LR, Sigmoid, linearOutput = true, LF, L2R))
  train(fixedNetwork)

  val x2 = -5.0 until 5.0 by 0.83
  val actual = x2.map(x => x -> linear(x))
  val linear = x2.map(x => x -> linearNetwork.feedForward(Array(x)).head)
  val overfitted = x2.map(x => x -> overfittedNetwork.feedForward(Array(x)).head)
  val fixed = x2.map(x => x -> fixedNetwork.feedForward(Array(x)).head)

  plot("Linear",
    Series("actual", actual),
    Series("linear", linear),
    Series("overfitted", overfitted),
    Series("fixed", fixed),
    Series("generated", generated, dots = true))

  def linear(x: Double): Double = 5 * x - 4

  def train(network: Network): Unit = {
    val losses = for (epoch <- 1 to 200) yield {
      SGD(network, 100, trainingData)
      val loss = evaluate(network, trainingData)
      logger.info(f"[$epoch]: loss: $loss%f")
      epoch.toDouble -> loss
    }
    plot("Linear", Series("loss", losses))
  }

}
