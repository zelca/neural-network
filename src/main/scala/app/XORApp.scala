package app

import neural.Network
import neural.function.Step

/**
  * A sample network built on perceptrons to solve XOR function
  */
object XORApp extends App {

  val biases = List(List(-0.5, 1.5), List(-1.5))
  val weights = List(List(Array(1.0, 1.0), Array(-1.0, -1.0)), List(Array(1.0, 1.0)))
  val network = Network(biases, weights, Step)

  assert(Array(0.0) sameElements network.feedForward(Array(0.0, 0.0)))
  assert(Array(1.0) sameElements network.feedForward(Array(1.0, 0.0)))
  assert(Array(1.0) sameElements network.feedForward(Array(0.0, 1.0)))
  assert(Array(0.0) sameElements network.feedForward(Array(1.0, 1.0)))

}
