# Spark-GBTLR
[![Build Status](https://travis-ci.org/titicaca/spark-gbtlr.svg?branch=master)](https://travis-ci.org/titicaca/spark-gbtlr)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


GBTLRClassifier is a hybrid model of Gradient Boosting Trees and Logistic Regression. 
It is quite practical and popular in many data mining competitions.
In this hybrid model, input features are transformed by means of boosted decision trees.
The output of each individual tree is treated as a categorical input feature to a sparse linear classifer. 
Boosted decision trees prove to be very powerful feature transforms.

Model details about GBTLR can be found in the following paper:
<a href="https://dl.acm.org/citation.cfm?id=2648589">Practical Lessons from Predicting Clicks on Ads at Facebook</a> [1].

GBTLRClassifier on Spark is designed and implemented by combining GradientBoostedTrees and Logistic Regressor in 
Spark MLlib. Features are firstly trained and transformed into sparse vectors via GradientBoostedTrees, and then
the generated sparse features will be trained and predicted in Logistic Regression model.

## Usage

GBTLRClassifier is designed and implemented easy to use. Parameters of GBTLRClassifier are the same as the combined 
parameters of GradientBoostedTrees and LogisticRegression in MLlib.

## Examples

The following codes are an example for predicting bank marketing results using Bank Marketing Dataset [2]. 
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).


*Scala API*
```scala
def main(args: Array[String]): Unit = {
    val spark = SparkSession
        .builder()
        .master("local[2]")
        .appName("gbtlr example")
        .getOrCreate()

    val startTime = System.currentTimeMillis()

    val dataset = spark.read.option("header", "true").option("inferSchema", "true")
        .option("delimiter", ";").csv("data/bank/bank-full.csv")

    val columnNames = Array("job", "marital", "education",
      "default", "housing", "loan", "contact", "month", "poutcome", "y")
    val indexers = columnNames.map(name => new StringIndexer()
        .setInputCol(name).setOutputCol(name + "_index"))
    val pipeline = new Pipeline().setStages(indexers)
    val data1 = pipeline.fit(dataset).transform(dataset)
    val data2 = data1.withColumnRenamed("y_index", "label")

    val assembler = new VectorAssembler()
    assembler.setInputCols(Array("age", "job_index", "marital_index",
      "education_index", "default_index", "balance", "housing_index",
      "loan_index", "contact_index", "day", "month_index", "duration",
      "campaign", "pdays", "previous", "poutcome_index"))
    assembler.setOutputCol("features")

    val data3 = assembler.transform(data2)
    val data4 = data3.randomSplit(Array(4, 1))

    val gBTLRClassifier = new GBTLRClassifier()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setGBTMaxIter(10)
        .setLRMaxIter(100)
        .setRegParam(0.01)
        .setElasticNetParam(0.5)

    val model = gBTLRClassifier.fit(data4(0))
    val summary = model.evaluate(data4(1))
    val endTime = System.currentTimeMillis()
    val auc = summary.binaryLogisticRegressionSummary
        .asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"Training and evaluating cost ${(endTime - startTime) / 1000} seconds")
    println(s"The model's auc: ${auc}")
```


## Benchmark
TO BE ADDED..

## Requirements

Spark-GBTLR is built on Spark 2.1.1 or later version.

## Build From Source

`mvn clean package`

## Licenses

Spark-GBTLR is available under Apache Licenses 2.0.

## Acknowledgement

Spark GBTLR is designed and implemented together with my former intern Fang, Jie at Transwarp (transwarp.io). 
Thanks for his great contribution. In addition, thanks for the supports of Discover Team.

## Contact and Feedback

If you encounter any bugs, feel free to submit an issue or pull request. Also you can email to:
<a href="fangzhou.yang@hotmail.com">Yang, Fangzhou (fangzhou.yang@hotmail.com)</a>


## References

[1] He X, Pan J, Jin O, et al. Practical Lessons from Predicting Clicks on Ads at Facebook[J]., 2014: 1-9.

[2] Moro S, Cortez P, Rita P, et al. A Data-Driven Approach to Predict the Success of Bank Telemarketing[J]. 
Decision support systems, 2014, 62(62): 22-31.
