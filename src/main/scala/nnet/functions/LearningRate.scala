package nnet.functions

object LearningRate {

  type LearningRate = (() => Double)

  def constant(rate: Double): LearningRate = () => rate

}