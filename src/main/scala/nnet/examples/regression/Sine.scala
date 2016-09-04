package nnet.examples.regression

import com.typesafe.scalalogging.Logger
import nnet.NetworkSpec._
import nnet.examples.utils._
import nnet.functions._
import nnet.{Network, NetworkSpec}
import org.slf4j.LoggerFactory

import scala.util.Random

object Sine extends App {

  val Epochs = 300
  val PointsCount = 100

  val LF = Quadratic
  val LR = LearningRate.constant(0.1)

  val x1 = Array.fill(PointsCount)(10 * Random.nextDouble())
  val generated = x1.map(x => x -> (math.sin(x) + 0.2 * Random.nextGaussian()))
  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))

  val x2 = Array.fill(PointsCount)(10 * Random.nextDouble())
  val validationData = x2.map(x => (Array(math.sin(x) + 0.2 * Random.nextGaussian()), Array(x)))

  val network = Network(NetworkSpec(List(1, 6, 1), LR, Sigmoid, linearOutput = true, LF))
  train(network, Epochs, trainingData, validationData)

  val net = hardcodedNetwork()

  val x3 = 0.0 until 10.0 by 0.1
  val actual = x3.map(x => x -> math.sin(x))
  val hardcoded = x3.map(x => x -> net.feedForward(Array(x)).head)
  val predicted = x3.map(x => x -> network.feedForward(Array(x)).head)

  plot("Sine",
    dots("generated", generated),
    line("actual", actual),
    line("hardcoded", hardcoded),
    line("predicted", predicted))

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

