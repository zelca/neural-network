package nnet.examples.mnist

import nnet.Network._

object MNISTData {

  private val TestingLabels = "mnist/t10k-labels.idx1-ubyte"

  private val TestingImages = "mnist/t10k-images.idx3-ubyte"

  private val TrainingLabels = "mnist/train-labels.idx1-ubyte"

  private val TrainingImages = "mnist/train-images.idx3-ubyte"

  def testing(): Seq[(Label, Vector)] = {
    MNISTDigit.load(TestingLabels, TestingImages).map(x => toInput(x._1, x._2))
  }

  def training(): Seq[(Label, Vector)] = {
    MNISTDigit.load(TrainingLabels, TrainingImages).map(x => toInput(x._1, x._2))
  }

  private def toInput(label: Int, image: Array[Byte]): Input = {
    (toLabel(label), image.map(x => (x & 0xFF).toDouble / 255.0))
  }

  private def toLabel(value: Int): Label = {
    (0 until 10).map(i => if (i == value) 1.0 else 0.0).toArray
  }

}
