package nnet.examples

import java.awt.{Color, Font}

import com.typesafe.scalalogging.Logger
import nnet.Network
import nnet.Network._
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer
import org.slf4j.LoggerFactory

import scalax.chart.module.ChartFactories.XYLineChart

package object utils {

  val logger = Logger(LoggerFactory.getLogger("utils"))

  def plot(title: String, series: Series*): Unit = {
    val data = series.map(s => s.title -> s.xy)
    val chart = XYLineChart(data, title = title)
    val font = chart.peer.getTitle.getFont
    chart.peer.getTitle.setFont(font.deriveFont(Font.PLAIN))
    chart.plot.setBackgroundPaint(Color.white)
    val renderer = chart.plot.getRenderer.asInstanceOf[XYLineAndShapeRenderer]
    for (i <- series.indices; if series(i).dots) {
      renderer.setSeriesShapesVisible(i, true)
      renderer.setSeriesLinesVisible(i, false)
    }
    chart.show()
  }

  case class Series(title: String, xy: Seq[(Double, Double)], dots: Boolean)

  def dots[T](title: String, xy: Seq[(T, Double)])(implicit n: Numeric[T]): Series =
    Series(title, xy.map(x => (n.toDouble(x._1), x._2)), dots = true)

  def line[T](title: String, xy: Seq[(T, Double)])(implicit n: Numeric[T]): Series =
    Series(title, xy.map(x => (n.toDouble(x._1), x._2)), dots = false)

  def train(network: Network, epochs: Int, trainingData: Seq[Input], validationData: Seq[Input]): Unit = {
    val errors = for (epoch <- 1 to epochs) yield {
      network.SGD(trainingData)
      val training = network.evaluate(trainingData)
      val validation = network.evaluate(validationData)
      logger.info(f"[$epoch]: error: $validation%f")
      epoch -> (training, validation)
    }
    val trainingLine = line("training error", errors.map(i => i._1 -> i._2._1))
    val validationLine = line("validation error", errors.map(i => i._1 -> i._2._2))
    plot("Errors", trainingLine, validationLine)
  }

}
