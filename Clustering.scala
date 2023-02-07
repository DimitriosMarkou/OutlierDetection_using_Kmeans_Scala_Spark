//package org.apache.spark.examples.mllib


import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import scala.io.Source
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vector
import org.apache.log4j.{Level, Logger}

object Clustering {

  def main(args: Array[String]): Unit = {

    Logger.getRootLogger().setLevel(Level.OFF) //Disable consol info.

    val spark = SparkSession.builder().appName("myApp").config("spark.master", "local").getOrCreate()

    // Read in the raw data from the input file.
    val rawData = Source.fromFile("raw_data.txt").getLines.toList
    //rawData.take(5).foreach(println)

    import spark.sqlContext.implicits._

    // Split the raw data into a list of lists, where each inner list represents a row of data.
    var data = rawData.map(_.split(",").toList)
    //data.take(5).foreach(println)

    //Missing data.
    val data_dropped = data.filter(row => row.size == 1 || row(0).isEmpty || row(1).isEmpty)

    //Filter the data to keep only those with 2 values.
    data = data.filter(row => row.size == 2) //When the left value is missing it still count it as length 2.
    data = data.filter(row => row(0).nonEmpty && row(1).nonEmpty) //Then check both the values are not empty.

    println("Data is Cleaned")
    println("Data excluded was")
    data_dropped.foreach(println)

    //Convert the data to a DataFrame with "x" column containing the first value and "y" column containing the second value.
    val rdd = spark.sparkContext.parallelize(data.map(r => (r(0), r(1))))
    var dataDF = rdd.toDF("x", "y")
    //val rdd = spark.sparkContext.parallelize(data.map(r => (r(0), r(1)))).zipWithIndex.map { case (v, i) => (i, v._1, v._2) }
    //var dataDF = rdd.toDF("id", "x", "y")

    //Convert strings to floats.
    dataDF = dataDF.withColumn("x", dataDF("x").cast("float"))
    dataDF = dataDF.withColumn("y", dataDF("y").cast("float"))

    println("")
    println("DataDF dataframe created.")
    dataDF.show(5,false)

   //Create a new column that is a Vector of the coordinates x,y
   val assembler = new VectorAssembler()
     .setInputCols(Array("x", "y"))
     .setOutputCol("features")

    dataDF = assembler.transform(dataDF)

    //Standarize the data.
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)

    dataDF = scaler.fit(dataDF).transform(dataDF)
    //dataDF.show(5,false)

    /**
    //Find the best K
    val sseList = scala.collection.mutable.ListBuffer.empty[Double]
    val silhouetteList = scala.collection.mutable.ListBuffer.empty[Double]
    val kValues = (100 to 800 by 25)
    val evaluator = new ClusteringEvaluator().setFeaturesCol("scaledFeatures")

    for (k <- kValues) {
      val kmeans = new KMeans().setK(k).setSeed(1L).setMaxIter(1000).setFeaturesCol("scaledFeatures")
      val model = kmeans.fit(dataDF)

      // Make predictions
      val transformed = model.transform(dataDF)
      val kmeansSummary = model.summary
      val sse = kmeansSummary.trainingCost
      val silhouette = evaluator.evaluate(transformed)

      sseList += sse
      silhouetteList += silhouette
    }
    println("")
    println("Scores from k=100 to k=500 with step=25")
    println("SSE: " + sseList)
    println("Silhouette: " + silhouetteList)
    */

    // K-means model.
    val k = 300
    val kmeans = new KMeans().setK(k).setSeed(1L).setMaxIter(1000).setFeaturesCol("scaledFeatures")

    //Train model
    val model = kmeans.fit(dataDF)

    // Make predictions
    dataDF = model.transform(dataDF)//.select("scaledFeatures")
    val kmeansSummary = model.summary
    val sse = kmeansSummary.trainingCost

    //kmeansSummary

    println("K-means model trained")
    dataDF.show(5,false)

    println("SSE")
    println(sse)
    println("***************************************************")

    //Evaluate clustering by computing Silhouette score
    val evaluator2 = new ClusteringEvaluator().setFeaturesCol("scaledFeatures")

    val silhouette = evaluator2.evaluate(dataDF)
    println(s"Silhouette with squared euclidean distance")
    println(silhouette)
    println("***************************************************")

    //Print the centers.
    println("")
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
    val clusterCenters = model.clusterCenters

    //For every data point we set the center of its cluster to a new column named centers.
    val matchCenter = udf((prediction: Int) => clusterCenters(prediction))
    dataDF = dataDF.withColumn("centers", matchCenter(dataDF("prediction")))
    //dataDF.show(5,false)

    //Calculate the distance of each data point from his center.
    val euclideanDistance = udf((v1: Vector, v2: Vector) => math.sqrt(Vectors.sqdist(v1, v2)))
    dataDF = dataDF.withColumn("distance", euclideanDistance($"scaledFeatures", $"centers"))
    //dataDF.show(5,false)

    /**
    // Save DataFrame to a CSV file
    val saveDF = dataDF.drop("features","scaledFeatures","centers","x","y") //Exclude some columns if it needs.
    saveDF.write
      .format("csv")
      .option("header","true")
      .mode("overwrite") // specify "overwrite" to overwrite the file if it already exists
      .save("path/name.csv")
    */

    println("")
    println("Outliers:")
    //Find the data point that the distance of its cluster is greter than a threshold.
    val outliers = dataDF.filter(dataDF("distance") > 0.2)
    outliers.show(10,false)


    println("***********************************************************************************************")
    spark.stop()

  }
}
