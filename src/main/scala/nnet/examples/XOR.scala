package nnet.examples

import nnet.{Network, NetworkSpec}

/**
  * A sample network built on perceptrons to solve XOR function
  */
object XOR extends App {

  val layers = List(
    List((-0.5, Array(1.0, 1.0)), (1.5, Array(-1.0, -1.0))),
    List((-1.5, Array(1.0, 1.0)))
  )

  val network = Network(NetworkSpec.perceptron(layers))

  assert(network.feedForward(Array(0.0, 0.0)) sameElements Array(0.0))
  assert(network.feedForward(Array(1.0, 0.0)) sameElements Array(1.0))
  assert(network.feedForward(Array(0.0, 1.0)) sameElements Array(1.0))
  assert(network.feedForward(Array(1.0, 1.0)) sameElements Array(0.0))

}
