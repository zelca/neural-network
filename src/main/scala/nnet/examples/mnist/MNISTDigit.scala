package nnet.examples.mnist

import java.io._
import java.nio.ByteBuffer

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.Random

object MNISTDigit {

  val logger = Logger(LoggerFactory.getLogger("mnist-digit"))

  private val ImageSize = (28, 28)

  private val LabelMagicNumber = 2049

  private val ImageMagicNumber = 2051

  def load(labelFile: String, imageFile: String): Seq[(Int, Array[Byte])] = {
    val labels = inputStream(labelFile)
    val images = inputStream(imageFile)

    val labelMagicNumber = readInt(labels)
    val imageMagicNumber = readInt(images)
    assert(labelMagicNumber == LabelMagicNumber)
    assert(imageMagicNumber == ImageMagicNumber)

    val labelNumber = readInt(labels)
    val imageNumber = readInt(images)
    assert(labelNumber == imageNumber)

    val rowNumber = readInt(images)
    val columnNUmber = readInt(images)
    assert((rowNumber, columnNUmber) == ImageSize)

    val digits = for (i <- 0 until labelNumber) yield (readByte(labels), readImage(images))

    logger.debug("3 samples from: " + imageFile)
    Random.shuffle(digits).take(3).foreach(display)

    labels.close()
    images.close()

    digits
  }

  private def readByte(stream: InputStream): Int = {
    stream.read()
  }

  private def readInt(stream: InputStream): Int = {
    val intBuffer = Array.ofDim[Byte](4)
    stream.read(intBuffer)
    ByteBuffer.wrap(intBuffer).getInt
  }

  private def readImage(stream: InputStream): Array[Byte] = {
    val imageBuffer = Array.ofDim[Byte](ImageSize._1 * ImageSize._2)
    stream.read(imageBuffer)
    imageBuffer
  }

  private def display(digit: (Int, Array[Byte])): Unit = {
    logger.debug("Raw sample: " + digit._1)
    digit._2.map(x => if (x == 0) ''' else 'X').grouped(28).map(_.mkString).foreach(r => logger.debug(r))
  }

  private def inputStream(file: String): InputStream = {
    new BufferedInputStream(getClass.getClassLoader.getResourceAsStream(file))
  }

}
