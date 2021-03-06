package nnet

import nnet.functions.{Activation, Regularization}

import scala.util.Random

class Neuron(activation: Activation, var b: Double, var w: Array[Double]) {

  var x = Array[Double]() // last input values

  var z = 0.0 // last weighted value

  var dc = 1.0 // dropout coefficient

  /**
    * @param input - output of the preceding layer
    * @return activation value = sum(weight[i] * input[i]) + bias
    */
  def forward(input: Array[Double]): Double = {
    assert(w.size == input.length, "Weights count must be equal to inputs count")
    if (dc == 0.0)
      dc
    else {
      x = input
      z = (w, x).zipped.map(_ * _).sum + b
      dc * activation(z)
    }
  }

  /**
    * Updates:
    * bias = bias - alpha * delta * activation'(z)
    * weight[i] = weight[i] - alpha * delta * activation'(z) * activation(value)
    *
    * @param delta  - delta, calculated based on subsequent layer
    * @param alpha  - learning rate
    * @param lambda - regularization
    * @return a list of deltas for every input = weight * delta * activation'(value)
    */
  def backward(delta: Double, alpha: Double, lambda: Regularization): Array[Double] = {
    if (dc == 0.0)
      w.map(dc * _)
    else {
      assert(w.size == x.length, "Feed forward must be executed first")
      val d = dc * delta * activation.gradient(z)
      val deltas = w.map(_ * d)
      b = b - alpha * d
      w = (w, x).zipped.map((w, x) => w - alpha * (d * x + lambda.gradient(w)))
      deltas
    }
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
    new Neuron(activation, nextGaussian(input), Array.fill(input)(nextGaussian(input)))
  }

  private def nextGaussian(size: Int): Double = Random.nextGaussian() / math.sqrt(size)

}
