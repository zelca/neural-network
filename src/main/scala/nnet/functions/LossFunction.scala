package nnet.functions

import nnet.Network.Label

trait LossFunction extends ((Label, Label) => Double) {

  /**
    * @param a - actual result
    * @param y - expected result
    * @return an array of deltas for each component of a label
    */
  def delta(a: Label, y: Label): Array[Double]

}

object Quadratic extends LossFunction {

  def apply(a: Label, y: Label): Double = {
    0.5 * (a, y).zipped.map {
      case (ai, yi) => math.pow(ai - yi, 2)
    }.sum
  }

  def delta(a: Label, y: Label): Array[Double] = {
    (a, y).zipped.map(_ - _)
  }

}

object Entropy extends LossFunction {

  val LowerLimit = 1.0e-80

  val UpperLimit = 1 - 1.0e-80

  def apply(a: Label, y: Label): Double = {
    -1 * (a, y).zipped.map {
      case (ai, yi) =>
        val ac = ai.max(LowerLimit).min(UpperLimit)
        yi * math.log(ac) + (1.0 - yi) * math.log(1.0 - ac)
    }.sum
  }

  def delta(a: Label, y: Label): Array[Double] = {
    (a, y).zipped.map {
      case (ai, yi) =>
        val ac = ai.max(LowerLimit).min(UpperLimit)
        (ac - yi) / (ac * (1.0 - ac))
    }
  }

}