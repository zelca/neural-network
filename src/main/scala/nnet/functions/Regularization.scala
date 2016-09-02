package nnet.functions

import nnet.Network

trait Regularization extends (Network => Double) {

  def gradient(w: Double): Double

}

/**
  * No regularization
  */
object NoRegularization extends Regularization {

  override def apply(n: Network): Double = 0

  override def gradient(w: Double): Double = 0

}

/**
  * Adds rate * sum(|weight|) to loss function
  * and -1 * rate * sign(weight) during weight update
  */
case class L1(rate: Double) extends Regularization {

  override def apply(n: Network): Double = {
    rate * n.layers.flatMap(_.flatMap(_.w)).map(math.abs).sum
  }

  override def gradient(w: Double): Double = {
    rate * math.signum(w)
  }

}

/**
  * Adds 0.5 * rate * sum(weight * weight) to loss function
  * and rate * weight during weight update
  */
case class L2(rate: Double) extends Regularization {

  override def apply(n: Network): Double = {
    0.5 * rate * n.layers.flatMap(_.flatMap(_.w)).map(math.pow(_, 2)).sum
  }

  override def gradient(w: Double): Double = {
    rate * w
  }

}

