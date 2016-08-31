package nnet.examples.regression

import nnet.examples.plotting._
import nnet.functions._
import nnet.{Network, NetworkSpec, _}

import scala.util.Random

object Sine extends App {

  val LF = Quadratic
  val LR = LearningRate.constant(0.1)

  val x1 = 0.0 until 10.0 by 0.05
  val actual = x1.map(x => x -> math.sin(x))
  val generated = x1.map(x => x -> (math.sin(x) + 0.1 * Random.nextGaussian()))

  val network = Network(NetworkSpec(Left(List(1, 6, 1)), LR, Sigmoid, linearOutput = true, LF))

  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))
  val losses = for (epoch <- 1 to 500) yield {
    SGD(network, 200, trainingData)
    val loss = evaluate(network, trainingData)
    logger.info(f"[$epoch]: loss: $loss%f")
    epoch.toDouble -> loss
  }
  plot("Sine", Series("loss", losses))

  val x2 = 0.0 until 10.0 by 0.03
  val net = hardcodedNetwork()
  val hardcoded = x2.map(x => x -> net.feedForward(Array(x)).head)
  val predicted = x2.map(x => x -> network.feedForward(Array(x)).head)

  plot("Sine",
    Series("actual", actual),
    Series("hardcoded", hardcoded),
    Series("predicted", predicted),
    Series("generated", generated, dots = true))

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

