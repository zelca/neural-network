package neural.function

trait Activation extends (Double => Double) {

  def gradient(z: Double): Double

}

object Step extends Activation {

  def apply(z: Double): Double =
    if (z > 0.0) 1.0 else 0.0

  def gradient(Z: Double): Double =
    throw new IllegalArgumentException("Has no gradient")

}