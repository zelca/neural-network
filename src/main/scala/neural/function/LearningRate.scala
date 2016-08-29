package neural.function

object LearningRate {

  type LearningRate = (() => Double)

  def constant(rate: Double): LearningRate = () => rate

}