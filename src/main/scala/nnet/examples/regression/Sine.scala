package nnet.examples.regression

import nnet.examples.plotting._
import nnet.functions._
import nnet.{Network, NetworkSpec, _}

import scala.util.Random

object Sine extends App {

  val LF = Quadratic
  val LR = LearningRate.constant(0.1)
  val L2R = new L2(0.01)

  val x1 = 0.0 until 10.0 by 1
  val generated = x1.map(x => x -> (math.sin(x) + 0.5 * Random.nextGaussian()))
  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))

  val overfittedNetwork = Network(NetworkSpec(List(1, 20, 1), LR, Sigmoid, linearOutput = true, LF))
  train(overfittedNetwork)

  val fixedNetwork = Network(NetworkSpec(List(1, 20, 1), LR, Sigmoid, linearOutput = true, LF, L2R))
  train(fixedNetwork)

  val x2 = 0.0 until 10.0 by 0.03
  val net = hardcodedNetwork()
  val actual = x2.map(x => x -> math.sin(x))
  val hardcoded = x2.map(x => x -> net.feedForward(Array(x)).head)
  val overfitted = x2.map(x => x -> overfittedNetwork.feedForward(Array(x)).head)
  val fixed = x2.map(x => x -> fixedNetwork.feedForward(Array(x)).head)

  plot("Sine",
    Series("actual", actual),
    Series("hardcoded", hardcoded),
    Series("overfitted", overfitted),
    Series("fixed", fixed),
    Series("generated", generated, dots = true))

  def train(network: Network): Unit = {
    val losses = for (epoch <- 1 to 1000) yield {
      SGD(network, 200, trainingData)
      val loss = evaluate(network, trainingData)
      logger.info(f"[$epoch]: loss: $loss%f")
      epoch.toDouble -> loss
    }
    // plot("Sine", Series("loss", losses))
  }

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

