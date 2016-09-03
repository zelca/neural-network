package nnet

import nnet.Network.Layer
import nnet.functions.{Activation, Linear}

class Network(val spec: NetworkSpec, val layers: Array[Layer]) {

  def getActivation(last: Boolean): Activation =
    if (last && spec.linearOutput) Linear else spec.activation

  def GD(data: Seq[Input]): Unit = {
    val learningRate = spec.learningRate.get
    val lossFunction = spec.lossFunction.get
    val regularization = spec.regularization
    val ii = Array.ofDim[Vector](layers.length)
    val zz = layers.map(nn => Array.ofDim[Double](nn.length))
    val aa = layers.map(nn => Array.ofDim[Double](nn.length))
    val dbb = layers.map(nn => Array.fill(nn.length)(0.0))
    val dww = layers.map(_.map(n => Array.fill(n.w.length)(0.0)))
    data.foreach {
      case (label, input) =>
        // feed forward
        var i = input
        for (l <- layers.indices; neurons = layers(l); a = aa(l); z = zz(l)) {
          ii(l) = i
          for (n <- neurons.indices; w = neurons(n).w; b = neurons(n).b) {
            z(n) = (w, i).zipped.map(_ * _).sum + b
            a(n) = getActivation(l == layers.length - 1)(z(n))
          }
          i = a
        }
        // calculate delta
        var dd = lossFunction.delta(i, label)
        // back propagation
        for (l <- layers.indices.reverse; neurons = layers(l);
             i = ii(l); a = aa(l); z = zz(l); db = dbb(l); dw = dww(l)) {
          val ndd = Array.ofDim[Vector](neurons.length)
          for (n <- neurons.indices; w = neurons(n).w; d = dd(n)) {
            val g = d * getActivation(l == layers.length - 1).gradient(z(n))
            db(n) += g
            for (k <- dw(n).indices) {
              dw(n)(k) = dw(n)(k) + i(k) * g
            }
            ndd(n) = w.map(_ * g)
          }
          dd = ndd.transpose.map(_.sum)
        }
    }
    // update
    for (l <- layers.indices; neurons = layers(l); db = dbb(l); dw = dww(l)) {
      for (n <- neurons.indices; w = neurons(n).w; b = neurons(n).b) {
        neurons(n).b -= learningRate() * db(n) / data.size
        for (k <- w.indices) {
          w(k) -= learningRate() * (dw(n)(k) + regularization.gradient(w(k)) ) / data.size
        }
      }
    }
  }

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