package neural.function

trait Activation extends (Double => Double) {

  def gradient(z: Double): Double

}

object UnitStep extends Activation {

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

  val UpperLimit = 100

  val LowerLimit = -100

  def apply(z: Double): Double = {
    if (z < LowerLimit)
      0.0
    else if (z > UpperLimit)
      1.0
    else
      1.0 / (1.0 + math.exp(-1 * z))
  }

  def gradient(z: Double): Double = {
    val sigmoid = Sigmoid(z)
    sigmoid * (1.0 - sigmoid)
  }

}