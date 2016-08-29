
import com.typesafe.scalalogging.Logger
import neural.function.LearningRate._
import neural.function.LossFunction
import org.slf4j.LoggerFactory

import scala.util.Random

package object neural {

  val logger = Logger(LoggerFactory.getLogger("neural"))

  type Label = Array[Double]

  type Vector = Array[Double]

  type Input = (Label, Vector)

  def SGD(network: Network, data: Seq[Input], batch: Int): Unit = {
    assert(network.spec.lossFunction.isDefined, "Loss function is required for SGD")
    val loss = network.spec.lossFunction.get
    Random.shuffle(data).take(batch).zipWithIndex.foreach {
      case ((label, features), index) =>
        val predicted = network.feedForward(features)
        val delta = loss.delta(predicted, label)
        network.backPropagation(delta)
        network.update()
        if ((index + 1) % 1000 == 0 || index == batch - 1)
          logger.debug("%s of %s processed".format(index + 1, batch))
    }
  }

  def evaluate(network: Network, data: Seq[Input], lossFunction: LossFunction): Double = {
    data.map(i => (network.feedForward(i._2), i._1)).map(p => lossFunction(p._1, p._2)).sum / data.size
  }

}
