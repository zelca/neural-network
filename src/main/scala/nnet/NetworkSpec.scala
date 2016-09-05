package nnet

import nnet.NetworkSpec.{LayersSpec, LearningRateSpec, LossFunctionSpec}
import nnet.functions.LearningRate.LearningRate
import nnet.functions.{Regularization, _}

class NetworkSpec(val layers: LayersSpec,
                  val activation: Activation,
                  val linearOutput: Boolean,
                  val lossFunction: LossFunctionSpec,
                  val learningRate: LearningRateSpec,
                  val regularization: Regularization,
                  val dropoutProbability: Double) {

  assert(activation != Linear || !linearOutput,
    "Activation function is already linear")

  assert(activation != UnitStep || learningRate.isEmpty,
    "Unit step function can not be used for learning")

  assert(dropoutProbability >= 0.0 && dropoutProbability < 1.0,
    "Dropout probability has to be between 0.0 and 1.0 (exclusive)")

  assert(!(lossFunction.contains(Entropy) && (activation != Sigmoid || linearOutput)),
    "Cross entropy is only used with Sigmoid activation for all layers")

  layers match {
    case Left(sizes) =>
      assert(sizes.size >= 2, "At least input and output layers must be defined")
    case Right(values) =>
      assert(values.nonEmpty, "Biases and weights for at least one layer must be defined")
  }

}

object NetworkSpec {

  type LearningRateSpec = Option[LearningRate]

  type LossFunctionSpec = Option[LossFunction]

  type LayersSpec = Either[List[Int], List[List[(Double, Array[Double])]]]

  implicit def anyToOptionAny[A](a: A): Option[A] = Some(a)

  implicit def anyToLeftAny[A, B](a: A): Either[A, B] = Left(a)

  def perceptron(layers: List[List[(Double, Array[Double])]]): NetworkSpec = {
    new NetworkSpec(Right(layers), UnitStep, false, None, None, NoRegularization, 0.0)
  }

  def linear(layers: LayersSpec, learningRate: LearningRate): NetworkSpec = {
    new NetworkSpec(layers, Linear, false, Quadratic, learningRate, NoRegularization, 0.0)
  }

  def apply(layers: LayersSpec,
            learningRate: LearningRateSpec,
            activation: Activation = Sigmoid,
            linearOutput: Boolean = false,
            lossFunction: LossFunctionSpec = Entropy,
            regularization: Regularization = NoRegularization,
            dropoutProbability: Double = 0.0): NetworkSpec = {
    new NetworkSpec(layers, activation, linearOutput, lossFunction, learningRate, regularization, dropoutProbability)
  }

}
