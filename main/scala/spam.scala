import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.lit


object spam {

  var metaRegex = Map[String, String]()

  metaRegex += ("digits" -> "\\b\\d+\\b")
  metaRegex += ("white_space" -> "\\s+")
  metaRegex += ("small_words" -> "\\b[a-zA-Z0-9]{1,2}\\b")
  metaRegex += ("urls" -> "(https?\\://)\\S+")
  metaRegex += ("currency" -> "[\\$\\€\\£]")

  def removeRegex(txt: String, flag: String): String = {
    val regex = metaRegex.get(flag)
    var cleaned = txt
    regex match {
      case Some(value) =>
        if (value.equals("white_space")) cleaned = txt.replaceAll(value, "")
        else if (value.equals("small_words")) cleaned = txt.replaceAll(value, "")
        else if (value.equals("urls")) cleaned = txt.replaceAll(value, "normalizedURL")
        else if (value.equals("digits")) cleaned = txt.replaceAll(value, "normalizedNUMBER")
        else if (value.equals("currency")) cleaned = txt.replaceAll(value, "normalizedCURRENCY")
        else  cleaned = txt.replaceAll(value, " ")
      case None => println("No regex flag matched")
    }
    cleaned
  }

  def cleanText(txt: String) : String = {
    var text = txt.toLowerCase
    text = removeRegex(text,"white_space")
    text = removeRegex(text,"small_words")
    text = removeRegex(text,"urls")
    text = removeRegex(text,"digits")
    text = removeRegex(text,"currency")
    text
  }

  def main(args: Array[String]): Unit = {

    // start spark session
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Classification")
      .getOrCreate()

    // load data as spark-datasets
    val spam_training = spark.read.textFile("src/main/resources/spam_training.txt")
    val spam_testing = spark.read.textFile("src/main/resources/spam_testing.txt")
    val nospam_training = spark.read.textFile("src/main/resources/nospam_training.txt")
    val nospam_testing = spark.read.textFile("src/main/resources/nospam_testing.txt")

    // implement: convert datasets to either rdds or dataframes (your choice) and build your pipeline
    val spam_trainingDF = spam_training.toDF("line").withColumn("label", lit(1))
    val spam_testingDF = spam_testing.toDF("line").withColumn("label", lit(1))
    val nospam_trainingDF = nospam_training.toDF("line").withColumn("label", lit(0))
    val nospam_testingDF = nospam_testing.toDF("line").withColumn("label", lit(0))

    val trainingDF = spam_trainingDF.union(nospam_trainingDF)
    val testDF = spam_testingDF.union(nospam_testingDF)

    val tokenizer = new Tokenizer().setInputCol("line").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val hashingTF = new HashingTF().setNumFeatures(2000).setInputCol("filtered").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.001).setLabelCol("label").setFeaturesCol("features")
    val nb = new NaiveBayes().setLabelCol("label").setFeaturesCol("features")

    val pipeline_lr = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, lr))

    val pipeline_nb = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, nb))

    val model_lr = pipeline_lr.fit(trainingDF)
    val prediction_lr =  model_lr.transform(testDF)
    val evaluator_lr = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy_lr = evaluator_lr.evaluate(prediction_lr)
    println("Test set accuracy LogisticRegression = " + accuracy_lr)

    val model_nb = pipeline_nb.fit(trainingDF)
    val predictions_nb = model_nb.transform(testDF)
    val evaluator_nb = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy_nb = evaluator_nb.evaluate(predictions_nb)
    println("Test set accuracy Naive Bayes= " + accuracy_nb)


  }
}
