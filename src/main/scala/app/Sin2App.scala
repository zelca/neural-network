package app

import com.typesafe.scalalogging.Logger
import neural.function.{Linear, Sigmoid}
import neural.{Input, Network}
import org.slf4j.LoggerFactory

import scala.util.Random

object SinApp extends App {

  val logger = Logger(LoggerFactory.getLogger("sin-app"))

  val biases = List(
    List(-1.55, -3.5, -8.4, -12.81, -0.49, 9.78),
    List(-2.59))

  val weights = List(
    List(Array(1.44), Array(-0.02), Array(0.93), Array(1.98), Array(1.84), Array(-3.14)),
    List(Array(-1.39, 3.85, -3.59, 2.92, 2.94, 1.70)))

  val network = Network(biases, weights, List(Sigmoid, Linear))

  for (i <- 0 until 10) {
    val sample = generate()
    val input = sample._2.head
    val expected = sample._1.head
    val predicted = network.feedForward(sample._2).head
    logger.info(f"$input%f: $expected%f == $predicted%f")
  }

  val sample = (Array(0.0), Array(0.0))
  val input = sample._2.head
  val expected = sample._1.head
  val predicted = network.feedForward(sample._2).head
  logger.info(f"$input%f: $expected%f == $predicted%f")

  val t = 1.55 * 1.39 + -3.5 * 3.85 + 8.4 * 3.59 - 12.81 * 2.92 - 0.49 * 2.94 + 9.78 * 1.70 - 2.59
  println(t)

  def generate(): Input = {
    val x = Random.nextDouble()
    val y = math.sin(x)
    (Array(y), Array(x))
  }

}
