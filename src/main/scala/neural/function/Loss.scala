package neural.function

import neural.Label

trait Loss extends ((Label, Label) => Double) {

  /**
    * @param a - actual result
    * @param y - expected result
    * @return an array of deltas for each component of a label
    */
  def delta(a: Label, y: Label): Array[Double]

}

object Quadratic extends Loss {

  def apply(a: Label, y: Label): Double =
    0.5 * (a, y).zipped.map {
      case (ai, yi) => math.pow(ai - yi, 2)
    }.sum

  def delta(a: Label, y: Label): Array[Double] =
    (a, y).zipped.map(_ - _)

}

object CrossEntropy extends Loss {

  def apply(a: Label, y: Label): Double =
    -1 * (a, y).zipped.map {
      case (ai, yi) => yi * math.log(ai) + (1.0 - yi) * math.log(1.0 - ai)
    }.sum

  def delta(a: Label, y: Label): Array[Double] =
    (a, y).zipped.map {
      case (ai, yi) => (ai - yi) / (ai * (1.0 - ai))
    }

}