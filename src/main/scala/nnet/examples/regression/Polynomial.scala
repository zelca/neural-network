package nnet.examples.regression

import com.typesafe.scalalogging.Logger
import nnet.NetworkSpec._
import nnet.examples.utils._
import nnet.functions._
import nnet.{Network, NetworkSpec}
import org.slf4j.LoggerFactory

import scala.util.Random

object Polynomial extends App {

  val Epochs = 500
  val PointsCount = 100

  val LF = Quadratic
  val LR = LearningRate.constant(0.01)

  val x1 = Array.fill(PointsCount)(-2.8 + 5.6 * Random.nextDouble())
  val generated = x1.map(x => x -> polynomialWithNoise(x))
  val trainingData = generated.map(xy => (Array(xy._2), Array(xy._1)))

  val x2 = Array.fill(PointsCount)(-2.8 + 5.6 * Random.nextDouble())
  val testingData = x2.map(x => (Array(polynomialWithNoise(x)), Array(x)))

  val network = Network(NetworkSpec(List(1, 10, 1), LR, linearOutput = true, lossFunction = LF))
  train(network, Epochs, trainingData, testingData)

  val x3 = -2.8 until 2.8 by 0.05
  val actual = x3.map(x => x -> polynomial(x))
  val predicted = x3.map(x => x -> network.feedForward(Array(x)).head)

  plot("Polynomial",
    dots("generated", generated),
    line("actual", actual),
    line("predicted", predicted))

  def polynomial(x: Double): Double = math.pow(x, 5) - 8 * math.pow(x, 3) + 10 * x + 6

  def polynomialWithNoise(x: Double): Double = polynomial(x) + Random.nextGaussian()

}

