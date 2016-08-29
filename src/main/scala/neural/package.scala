
import com.typesafe.scalalogging.Logger
import neural.function.Learning._
import neural.function.Loss
import org.slf4j.LoggerFactory

import scala.util.Random

package object neural {

  val logger = Logger(LoggerFactory.getLogger("neural"))

  type Label = Array[Double]

  type Features = Array[Double]

  type Input = (Label, Features)

  def stochasticGradientDescent(network: Network,
                                data: Seq[Input],
                                batch: Int,
                                loss: Loss,
                                learning: Learning): Unit = {
    Random.shuffle(data).take(batch).zipWithIndex.foreach {
      case ((label, features), index) =>
        val predicted = network.feedForward(features)
        network.backPropagation(loss.delta(predicted, label))
        network.update(learning())
        if ((index + 1) % 1000 == 0 || index == data.size - 1)
          logger.debug("%s of %s processed".format(index + 1, data.size))
    }
  }

}
