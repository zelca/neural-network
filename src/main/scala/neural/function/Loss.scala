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

  val MinLimit = 1.0e-80

  val MaxLimit = 1 - 1.0e-80

  def apply(a: Label, y: Label): Double =
    -1 * (a, y).zipped.map {
      case (ai, yi) =>
        val ac = ai.max(MinLimit).min(MaxLimit)
        yi * math.log(ac) + (1.0 - yi) * math.log(1.0 - ac)
    }.sum

  def delta(a: Label, y: Label): Array[Double] =
    (a, y).zipped.map {
      case (ai, yi) =>
        val ac = ai.max(MinLimit).min(MaxLimit)
        (ai - yi) / (ai * (1.0 - ai))
    }

}