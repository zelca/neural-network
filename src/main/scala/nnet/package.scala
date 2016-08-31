
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.Random

package object nnet {

  val logger = Logger(LoggerFactory.getLogger("neural"))

  type Label = Array[Double]

  type Vector = Array[Double]

  type Input = (Label, Vector)

  def SGD(network: Network, batch: Int, data: Seq[Input]): Unit = {
    assert(network.spec.lossFunction.isDefined, "Loss function is required for SGD")
    val lossFunction = network.spec.lossFunction.get
    val chunk = Random.shuffle(data).take(batch)
    for (i <- chunk.indices; (label, features) = chunk(i)) {
      val predicted = network.feedForward(features)
      network.backPropagation(lossFunction.delta(predicted, label))
      if (i % 1000 == 0) logger.debug("%s of %s processed".format(i + 1, chunk.size))
    }
  }

  def evaluate(network: Network, data: Seq[Input]): Double = {
    assert(network.spec.lossFunction.isDefined, "Loss function is required for evaluation")
    val lossFunction = network.spec.lossFunction.get
    data.map(i => (network.feedForward(i._2), i._1)).map(p => lossFunction(p._1, p._2)).sum / data.size
  }

}
