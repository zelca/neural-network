package neural.function

trait Activation extends (Double => Double) {

  def gradient(z: Double): Double

}

object Step extends Activation {

  def apply(z: Double): Double = {
    if (z > 0.0) 1.0 else 0.0
  }

  def gradient(Z: Double): Double = {
    throw new IllegalStateException("Has no gradient")
  }

}

object Identity extends Activation {

  def apply(z: Double): Double = {
    z
  }

  def gradient(z: Double): Double = {
    1.0
  }

}

object Sigmoid extends Activation {

  def apply(z: Double): Double = {
    1.0 / (1.0 + math.exp(-1 * z))
  }

  def gradient(z: Double): Double = {
    val sigmoid = Sigmoid(z)
    sigmoid * (1.0 - sigmoid)
  }

}