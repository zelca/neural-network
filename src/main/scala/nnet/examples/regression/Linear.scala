package nnet.examples.regression

import nnet.NetworkSpec._
import nnet.functions._
import nnet.{Network, NetworkSpec}
import nnet.examples.utils._

import scala.util.Random

object Linear extends App {

  val Epochs = 10
  val PointsCount = 20

  val LR = LearningRate.constant(0.001)

  val x1 = Array.fill(PointsCount)(10 * Random.nextDouble())
  val generated = x1.map(x => x -> linearWithNoise(x))
  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))

  val x2 = Array.fill(PointsCount)(10 * Random.nextDouble())
  val testingData = x2.map(x => (Array(linearWithNoise(x)), Array(x)))

  val network = Network(NetworkSpec.linear(List(1, 1), LR))
  train(network, Epochs, trainingData, testingData)

  val x3 = 0.0 until 10.0 by 0.1
  val actual = x3.map(x => x -> linear(x))
  val linear = x3.map(x => x -> network.feedForward(Array(x)).head)

  plot("Linear",
    dots("generated", generated),
    line("actual", actual),
    line("linear", linear))

  def linear(x: Double): Double = 5 * x - 4

  def linearWithNoise(x: Double): Double = linear(x) + 5 * Random.nextGaussian()

}
