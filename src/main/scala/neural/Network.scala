package neural

import neural.Network.Layer
import neural.function.Activation

class Network(layers: Array[Layer]) {

  /**
    * @param input - features of the training/testing sample
    * @return output of the last layer
    */
  def feedForward(input: Array[Double]): Array[Double] = {
    layers.foldLeft(input) {
      (input, layer) => layer.map(_.forward(input))
    }
  }

  /**
    * Propagates delta to all neurons in all layers
    *
    * @param delta - delta, calculated by loss function
    */
  def backPropagation(delta: Array[Double]): Unit = {
    layers.reverse.foldLeft(delta) {
      (delta, layer) =>
        assert(layer.length == delta.length)
        (layer, delta).zipped.map(_.backward(_)).transpose.map(_.sum)
    }
  }

  /**
    * Updates biases and weights for all neurons in all layers
    *
    * @param alpha - learning rate
    */
  def update(alpha: Double): Unit = {
    layers.par.foreach(_.par.foreach(_.update(alpha)))
  }

}

object Network {

  type Layer = Array[Neuron]

  /**
    * @param layout     - number of neurons for each layer
    * @param activation - activation function for all neurons
    * @return a network where neurons initialized with random (Gaussian) biases and weights
    */
  def apply(layout: List[Int], activation: Activation): Network = {
    assert(layout.size >= 2)
    val layers = layout.sliding(2).collect {
      case prev :: curr :: Nil => Array.fill(curr)(Neuron(activation, prev))
    }
    new Network(layers.toArray)
  }

  /**
    *
    * @param biases     - biases for each neuron in each layer
    * @param weights    - weights for each neuron in each layer
    * @param activation - activation function for all neurons
    * @return a network where neurons initialized with the given biases and weights
    */
  def apply(biases: List[List[Double]], weights: List[List[Array[Double]]], activation: Activation): Network = {
    assert(biases.size == weights.size)
    val layers = (biases, weights).zipped.map {
      (bl, wl) =>
        assert(bl.size == wl.size)
        (bl, wl).zipped.map((bn, wn) => Neuron(activation, bn, wn)).toArray
    }
    new Network(layers.toArray)
  }

}