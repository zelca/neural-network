package nnet.examples

import java.awt.{Color, Font}

import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer

import scalax.chart.module.Charting

package object plotting extends Charting {

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

}
