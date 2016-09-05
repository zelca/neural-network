package nnet.examples.mnist

import com.typesafe.scalalogging.Logger
import nnet.Network.{Input, Label}
import nnet.NetworkSpec._
import nnet.examples.utils._
import nnet.functions._
import nnet.{Network, NetworkSpec}
import org.slf4j.LoggerFactory

import scala.util.Random

/**
  * hidden: 100, epochs: 100, batch: 10000, learning: 0.03, l2 regularization: 0.00001, dropouts: 0.5
  * error: 0.038133 final error: 0.038597 matches: 95.40% final matches: 95.43%
  */
object MNIST extends App {

  val logger = Logger(LoggerFactory.getLogger("mnist"))

  val Epochs = 10
  val BatchSize = 10000
  val LF = Quadratic
  val LR = LearningRate.constant(0.1)
  val L2R = L2(0.00001)

  val testingData = Random.shuffle(MNISTData.testing())
  logger.info("Testing set size: " + testingData.size)
  logger.debug("3 samples of testing data:")
  display(testingData, 3)
  val testingSample = testingData.take(1000)

  val trainingData = MNISTData.training()
  logger.info("Training set size: " + trainingData.size)
  logger.debug("3 samples of training data:")
  display(trainingData, 3)
  val trainingSample = trainingData.take(1000)

  val sample = trainingData.head
  val layers = List(sample._2.length, 30, sample._1.length)
  val network = Network(NetworkSpec(layers, LR, lossFunction = LF, regularization = L2R))

  logger.info("%s iterations with %s samples".format(Epochs, BatchSize))
  val errors = for (epoch <- 1 to Epochs) yield {
    network.SGD(Random.shuffle(trainingData).take(BatchSize))
    val (testingError, testingMatches) = evaluate(network, testingSample)
    val (trainingError, trainingMatches) = evaluate(network, trainingSample)
    logger.info(f"******* $epoch *******")
    logger.info(f"testing error: $testingError%f")
    logger.info(f"testing matches: $testingMatches%1.2f%%")
    logger.info(f"training error: $trainingError%f")
    logger.info(f"training matches: $trainingMatches%1.2f%%")
    epoch -> (trainingError, testingError)
  }
  val testingLine = line("testing error", errors.map(i => i._1 -> i._2._2))
  val trainingLine = line("training error", errors.map(i => i._1 -> i._2._1))
  plot("Errors", trainingLine, testingLine)

  val (error, matches) = evaluate(network, testingData)
  logger.info(f"******* Final *******")
  logger.info(f"testing error: $error%f")
  logger.info(f"testing matches: $matches%1.2f%%")

  def toInt: Label => Int = _.zipWithIndex.max._2

  def evaluate(network: Network, data: Seq[Input]): (Double, Double) = {
    val lossFunction = network.spec.lossFunction.get
    val predicted = data.map(i => (network.feedForward(i._2), i._1))
    val error = predicted.map(p => lossFunction(p._1, p._2)).sum / data.size
    val matches = 100 * predicted.count(p => toInt(p._1) == toInt(p._2)).toDouble / data.size
    (error, matches)
  }

  def display(data: Seq[Input], count: Int): Unit = {
    List.fill(count)(Random.nextInt(data.size)).map(data(_)).foreach {
      input =>
        logger.debug("Input sample: " + toInt(input._1))
        input._2.map(x => if (x == 0) ''' else 'X').grouped(28).map(_.mkString).foreach(r => logger.debug(r))
    }
  }

}

