package nnet

import nnet.Network.Layer
import nnet.functions.{Activation, Linear}

class Network(val spec: NetworkSpec, layers: Array[Layer]) {

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
    assert(spec.learningRate.isDefined, "Learning rate is not defined")
    val alpha = spec.learningRate.get()
    layers.reverse.foldLeft(delta) {
      (delta, layer) =>
        assert(layer.length == delta.length)
        (layer, delta).zipped.map(_.backward(_, alpha)).transpose.map(_.sum)
    }
  }

}

object Network {

  type Layer = Array[Neuron]

  /**
    * @param spec - a valid network specification
    * @return a network in accordance with specification
    */
  def apply(spec: NetworkSpec): Network = {
    def getActivation(last: Boolean): Activation =
      if (last && spec.linearOutput) Linear else spec.activation

    val layers = spec.layers match {
      case Left(sizes) =>
        for (i <- 1 until sizes.size; prev = sizes(i - 1); curr = sizes(i); last = i == sizes.size - 1) yield {
          Array.fill(curr)(Neuron(getActivation(last), prev))
        }
      case Right(values) =>
        for (i <- values.indices; layer = values(i); last = i == values.size - 1) yield {
          layer.toArray.map {
            case (bias, weights) => Neuron(getActivation(last), bias, weights)
          }
        }
    }
    new Network(spec, layers.toArray)
  }

}