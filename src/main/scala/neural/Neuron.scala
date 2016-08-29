package neural

import neural.function.Activation

import scala.util.Random

class Neuron(activation: Activation, var b: Double, var w: Array[Double]) {

  var x = Array[Double]() // last input values

  var z = 0.0 // last weighted value

  var a = 0.0 // last activation value

  var d = 0.0 // last delta value

  /**
    * @param input - output of the preceding layer
    * @return activation value = sum(weight[i] * input[i]) + bias
    */
  def forward(input: Array[Double]): Double = {
    assert(w.size == input.length)
    x = input
    z = (w, x).zipped.map(_ * _).sum + b
    a = activation(z)
    a
  }

  /**
    * @param delta - delta, calculated based on subsequent layer
    * @return a list of deltas for every input = weight * delta * activation'(value)
    */
  def backward(delta: Double): Array[Double] = {
    d = delta * activation.gradient(z)
    w.map(_ * d)
  }

  /**
    * Updates:
    * bias = bias - alpha * delta
    * weight[i] = weight[i] - alpha * delta * activation(value)
    *
    * @param alpha - learning rate
    */
  def update(alpha: Double): Unit = {
    b = b - alpha * d
    w = (w, x).zipped.map(_ - alpha * d * _)
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
