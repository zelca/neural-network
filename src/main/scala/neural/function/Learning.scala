package neural.function

object Learning {

  type Learning = (() => Double)

  def constant(rate: Double): Learning = () => rate

}