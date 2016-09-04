package nnet

import com.typesafe.scalalogging.Logger
import nnet.Network.{Input, Layer}
import nnet.functions.{Activation, Linear}
import org.slf4j.LoggerFactory

class Network(val spec: NetworkSpec, val layers: Array[Layer]) {

  private val logger = Logger(LoggerFactory.getLogger("network"))

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
    val lambda = spec.regularization
    layers.reverse.foldLeft(delta) {
      (delta, layer) =>
        assert(layer.length == delta.length)
        (layer, delta).zipped.map(_.backward(_, alpha, lambda)).transpose.map(_.sum)
    }
  }

  /**
    * Predicts for the given points and calculates total loss
    *
    * @param data - dataset to predict on
    * @return total loss for  the given dataset
    */

  def evaluate(data: Seq[Input]): Double = {
    assert(spec.lossFunction.isDefined, "Loss function is required for evaluation")
    val lossFunction = spec.lossFunction.get
    val errors = data.map {
      case (label, features) => lossFunction(feedForward(features), label)
    }
    errors.sum / data.size + spec.regularization(this)
  }

  /**
    * Trains network using online stochastic gradient descent algorithm
    *
    * @param data - training dataset
    */
  def SGD(data: Seq[Input]): Unit = {
    assert(spec.lossFunction.isDefined, "Loss function is required for SGD")
    val lossFunction = spec.lossFunction.get
    for (i <- data.indices; (label, features) = data(i)) {
      val predicted = feedForward(features)
      backPropagation(lossFunction.delta(predicted, label))
      if (i % 1000 == 0) logger.debug("%s of %s processed".format(i + 1, data.size))
    }
  }

}

object Network {

  type Layer = Array[Neuron]

  type Label = Array[Double]

  type Vector = Array[Double]

  type Input = (Label, Vector)

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