package nnet

import nnet.NetworkSpec._
import nnet.functions.{LearningRate, Quadratic, Sigmoid}
import org.scalatest.{FunSuite, Matchers}

class NetworkSuite extends FunSuite with Matchers {

  val Epsilon = 1e-5

  test("Neuron feed forward") {
    val neuron = Neuron(Sigmoid, 0.35, Array(0.15, 0.20))
    val input = Array(0.05, 0.10)

    val actual = neuron.forward(input)
    val expected = 0.59327

    actual should be(expected +- Epsilon)
  }

  test("Network feed forward") {
    val net = network()
    val input = Array(0.05, 0.10)

    val actual = net.feedForward(input)
    val expected = Array(0.75136, 0.77293)

    actual.length should be(expected.length)
    for (i <- actual.indices) actual(i) should be(expected(i) +- Epsilon)
  }

  test("Quadratic loss") {
    val actual = Quadratic(Array(0.75136, 0.77293), Array(0.01, 0.99))
    val expected = 0.29837

    actual should be(expected +- Epsilon)
  }

  test("Quadratic loss delta") {
    val actual = Quadratic.delta(Array(0.75136, 0.77293), Array(0.01, 0.99))
    val expected = Array(0.74136, -0.21707)

    actual.length should be(expected.length)
    for (i <- actual.indices) actual(i) should be(expected(i) +- Epsilon)
  }

  test("Network back propagation") {
    val net = network()
    val input = Array(0.05, 0.10)
    val output = Array(0.01, 0.99)

    val delta = Quadratic.delta(net.feedForward(input), output)
    net.backPropagation(delta)

    val actual = Quadratic(net.feedForward(input), output)
    val expected = 0.280471

    actual should be(expected +- Epsilon)
  }

  private def network(): Network = {
    val layers = List(
      List((0.35, Array(0.15, 0.20)), (0.35, Array(0.25, 0.30))),
      List((0.60, Array(0.40, 0.45)), (0.60, Array(0.50, 0.55)))
    )

    val learningRate = LearningRate.constant(0.5)

    Network(NetworkSpec(Right(layers), learningRate, Sigmoid, linearOutput = false, Quadratic))
  }

}
