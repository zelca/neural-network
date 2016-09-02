package nnet

import nnet.functions.{Activation, Regularization}

import scala.util.Random

class Neuron(activation: Activation, var b: Double, var w: Array[Double]) {

  var x = Array[Double]() // last input values

  var z = 0.0 // last weighted value

  var a = 0.0 // last activation value

  /**
    * @param input - output of the preceding layer
    * @return activation value = sum(weight[i] * input[i]) + bias
    */
  def forward(input: Array[Double]): Double = {
    assert(w.size == input.length, "Weights count must be equal to inputs count")
    x = input
    z = (w, x).zipped.map(_ * _).sum + b
    a = activation(z)
    a
  }

  /**
    * Updates:
    * bias = bias - alpha * delta * activation'(z)
    * weight[i] = weight[i] - alpha * delta * activation'(z) * activation(value)
    *
    * @param delta  - delta, calculated based on subsequent layer
    * @param alpha  - learning rate
    * @param lamdba - regularization
    * @return a list of deltas for every input = weight * delta * activation'(value)
    */
  def backward(delta: Double, alpha: Double, lamdba: Regularization): Array[Double] = {
    assert(w.size == x.length, "Feed forward must be executed first")
    val d = delta * activation.gradient(z)
    val v = w.map(_ * d)
    b = b - alpha * d
    w = (w, x).zipped.map((w, x) => w - alpha * (d * x + lamdba.gradient(w)))
    v
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
