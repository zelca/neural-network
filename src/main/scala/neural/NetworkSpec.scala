package neural

import neural.NetworkSpec.{LayersSpec, LearningRateSpec, LossSpec}
import neural.function.LearningRate.LearningRate
import neural.function._

class NetworkSpec(val layers: LayersSpec,
                  val activation: Activation,
                  val linearOutput: Boolean,
                  val lossFunction: LossSpec,
                  val learningRate: LearningRateSpec) {

  assert(!(activation == Linear && linearOutput),
    "Activation function is already linear")

  assert(!(activation == UnitStep && learningRate.isDefined),
    "Unit step function can not be used for learning")

  assert(lossFunction.isDefined == learningRate.isDefined,
    "Both loss function and learning rate must be defined/undefined")

  assert(!(lossFunction.contains(CrossEntropy) && (activation != Sigmoid || linearOutput)),
    "Cross entropy is only used with Sigmoid activation for all layers")

  layers match {
    case Left(sizes) =>
      assert(sizes.size >= 2, "At least input and output layers must be defined")
    case Right(values) =>
      assert(values.nonEmpty, "Biases and weights for at least one layer must be defined")
  }

}

object NetworkSpec {

  type LossSpec = Option[LossFunction]

  type LearningRateSpec = Option[LearningRate]

  type LayersSpec = Either[List[Int], List[List[(Double, Array[Double])]]]

  def linear(layers: LayersSpec, learningRate: LearningRate): NetworkSpec = {
    new NetworkSpec(layers, Linear, false, Some(Quadratic), Some(learningRate))
  }

  def perceptron(layers: List[List[(Double, Array[Double])]]): NetworkSpec = {
    new NetworkSpec(Right(layers), UnitStep, false, None, None)
  }

  def apply(layers: LayersSpec,
            learningRate: LearningRate,
            activation: Activation = Sigmoid,
            linearOutput: Boolean = false,
            lossFunction: LossFunction = CrossEntropy): NetworkSpec = {
    new NetworkSpec(layers, activation, linearOutput, Some(lossFunction), Some(learningRate))
  }

}
