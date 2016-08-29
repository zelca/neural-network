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

object Linear extends Activation {

  def apply(z: Double): Double = {
    z
  }

  def gradient(z: Double): Double = {
    1.0
  }

}

object ReLU extends Activation {

  def apply(z: Double): Double = {
    z.max(0.0)
  }

  def gradient(z: Double): Double = {
    if (z > 0.0) 1.0 else 0.0
  }

}

object Sigmoid extends Activation {

  val MaxLimit = 100

  val MinLimit = -100

  def apply(z: Double): Double = {
    if (z < MinLimit)
      0.0
    else if (z > MaxLimit)
      1.0
    else
      1.0 / (1.0 + math.exp(-1 * z))
  }

  def gradient(z: Double): Double = {
    val sigmoid = Sigmoid(z)
    sigmoid * (1.0 - sigmoid)
  }

}