package neural

import neural.function.Activation

import scala.util.Random

class Neuron(activation: Activation, var b: Double, var w: Array[Double]) {

  /**
    * @param input output of the preceding layer
    * @return activation value = Sum(weight[i] * input[i]) + bias
    */
  def forward(input: Array[Double]): Double = {
    assert(w.size == input.length)
    val z = (w, input).zipped.map(_ * _).sum + b
    activation(z)
  }

}

object Neuron {

  /**
    * @return a neuron with the given activation function, bias and weights
    */
  def apply(activation: Activation, bias: Double, weights: Array[Double]): Neuron = {
    new Neuron(activation, bias, weights)
  }

  /**
    * @return a neuron with the given activation function and random (Gaussian) bias and weights
    */
  def apply(activation: Activation, input: Int): Neuron = {
    new Neuron(activation, Random.nextGaussian(), Array.fill(input)(Random.nextGaussian()))
  }

}
