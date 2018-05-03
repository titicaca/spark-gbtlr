package org.apache.spark.ml.gbtlr

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{LabeledPoint, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.linalg.{DenseVector => OldDenseVector}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, Node => OldNode}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._


class GBTLRClassifierSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  @transient var dataset: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    dataset = spark.createDataFrame(GBTLRClassifierSuite.generateOrderedLabeledPoints(10, 100))
  }

  test("default params") {
    val gBTLRClassifier = new GBTLRClassifier()

    assert(gBTLRClassifier.getSeed === gBTLRClassifier.getClass.getName.hashCode.toLong)
    assert(gBTLRClassifier.getSubsamplingRate === 1.0)
    assert(gBTLRClassifier.getGBTMaxIter === 20)
    assert(gBTLRClassifier.getStepSize === 0.1)
    assert(gBTLRClassifier.getMaxDepth === 5)
    assert(gBTLRClassifier.getMaxBins === 32)
    assert(gBTLRClassifier.getMinInstancePerNode === 1)
    assert(gBTLRClassifier.getMinInfoGain === 0.0)
    assert(gBTLRClassifier.getCheckpointInterval === 10)
    assert(gBTLRClassifier.getFitIntercept === true)
    assert(gBTLRClassifier.getProbabilityCol === "probability")
    assert(gBTLRClassifier.getRawPredictionCol === "rawPrediction")
    assert(gBTLRClassifier.getStandardization === true)
    assert(gBTLRClassifier.getThreshold === 0.5)
    assert(gBTLRClassifier.getLossType === "logistic")
    assert(gBTLRClassifier.getCacheNodeIds === false)
    assert(gBTLRClassifier.getMaxMemoryInMB === 256)
    assert(gBTLRClassifier.getRegParam === 0.0)
    assert(gBTLRClassifier.getElasticNetParam === 0.0)
    assert(gBTLRClassifier.getFamily === "auto")
    assert(gBTLRClassifier.getLRMaxIter === 100)
    assert(gBTLRClassifier.getTol === 1E-6)
    assert(gBTLRClassifier.getAggregationDepth === 2)
  }

  test("set params") {
    val gBTLRClassifier = new GBTLRClassifier()
        .setSeed(123L)
        .setSubsamplingRate(0.5)
        .setGBTMaxIter(10)
        .setStepSize(0.5)
        .setMaxDepth(10)
        .setMaxBins(20)
        .setMinInstancesPerNode(2)
        .setMinInfoGain(1.0)
        .setCheckpointInterval(5)
        .setFitIntercept(false)
        .setProbabilityCol("test_probability")
        .setRawPredictionCol("test_rawPrediction")
        .setStandardization(false)
        .setThreshold(1.0)
        .setCacheNodeIds(true)
        .setMaxMemoryInMB(128)
        .setRegParam(1.0)
        .setElasticNetParam(0.5)
        .setFamily("binomial")
        .setLRMaxIter(50)
        .setTol(1E-3)
        .setAggregationDepth(3)

    assert(gBTLRClassifier.getSeed === 123L)
    assert(gBTLRClassifier.getSubsamplingRate === 0.5)
    assert(gBTLRClassifier.getGBTMaxIter === 10)
    assert(gBTLRClassifier.getStepSize === 0.5)
    assert(gBTLRClassifier.getMaxDepth === 10)
    assert(gBTLRClassifier.getMaxBins === 20)
    assert(gBTLRClassifier.getMinInstancePerNode === 2)
    assert(gBTLRClassifier.getMinInfoGain === 1.0)
    assert(gBTLRClassifier.getCheckpointInterval === 5)
    assert(gBTLRClassifier.getFitIntercept === false)
    assert(gBTLRClassifier.getProbabilityCol === "test_probability")
    assert(gBTLRClassifier.getRawPredictionCol === "test_rawPrediction")
    assert(gBTLRClassifier.getStandardization === false)
    assert(gBTLRClassifier.getThreshold === 1.0)
    assert(gBTLRClassifier.getCacheNodeIds === true)
    assert(gBTLRClassifier.getMaxMemoryInMB === 128)
    assert(gBTLRClassifier.getRegParam === 1.0)
    assert(gBTLRClassifier.getElasticNetParam === 0.5)
    assert(gBTLRClassifier.getFamily === "binomial")
    assert(gBTLRClassifier.getLRMaxIter === 50)
    assert(gBTLRClassifier.getTol === 1E-3)
    assert(gBTLRClassifier.getAggregationDepth === 3)
  }

  test("combination features") {
    val gBTLRClassifier = new GBTLRClassifier()
    val model = gBTLRClassifier.train(dataset)
    val gbtModel = model.gbtModel
    var numLeafNodes = 0
    for (i <- 0 until gbtModel.trees.length) {
      numLeafNodes += (gbtModel.trees(i).numNodes + 1) / 2
    }
    val newDataset = model.summary.newDataset
    val numFeatures = model.lrModel.numFeatures
    val lrSummary = model.summary.logRegSummary
    val originNumFeatures = dataset.select(col(lrSummary.labelCol),
      col("features")).rdd.map {
      case Row(label: Double, features: Vector) => features
    }.first().size
    assert(numFeatures === originNumFeatures + numLeafNodes)
  }

  test("add features") {
    val gBTLRClassifier = new GBTLRClassifier()
    val model = gBTLRClassifier.train(dataset)
    val point = new OldLabeledPoint(1.0, new OldDenseVector(Array.fill(100)(0)))
    val combinedPoint = model.getComibinedFeatures(point)
    val combinedFeatures = combinedPoint.features.toArray
    assert(combinedFeatures.sum === 20.0)
  }

  test("rules") {
    val gBTLRClassifier = new GBTLRClassifier()
    val model = gBTLRClassifier.fit(dataset)
    val numTrees = model.gbtModel.trees.length
    var totalLeafNodes = 0
    for (i <- 0 until numTrees) {
      totalLeafNodes += (model.gbtModel.trees(i).numNodes + 1) / 2
    }
    assert(model.getRules.length === totalLeafNodes)
  }

// Uncomment to run GBTLRClassificationModel read / write tests
//  test("read/write") {
//
//    def checkModelData(model1: GBTLRClassificationModel, model2: GBTLRClassificationModel): Unit = {
//      assert(model1.gbtModel.algo === model2.gbtModel.algo)
//      try {
//        model1.gbtModel.trees.zip(model2.gbtModel.trees).foreach {
//          case (tree1, tree2) => checkModelEqual(tree1, tree2)
//        }
//        assert(model1.gbtModel.treeWeights === model2.gbtModel.treeWeights)
//      } catch {
//        case ex: Exception =>
//          throw new AssertionError("checkModelData failed since " +
//              "the two gbtModels were not identical.\n")
//      }
//      assert(model1.lrModel.intercept === model2.lrModel.intercept)
//      assert(model1.lrModel.coefficients.toArray === model2.lrModel.coefficients.toArray)
//      assert(model1.lrModel.numFeatures === model2.lrModel.numFeatures)
//      assert(model1.lrModel.numClasses === model2.lrModel.numClasses)
//    }
//
//    val gBTLRClassifier = new GBTLRClassifier()
//    testEstimatorAndModelReadWrite(
//      gBTLRClassifier, dataset,
//      GBTLRClassifierSuite.allParamSettings,
//      GBTLRClassifierSuite.allParamSettings,
//      checkModelData
//    )
//  }

  /**
    * Return true iff the two nodes and their descendents are exactly the same.
    */
  private def checkTreeEqual(a: OldNode, b: OldNode): Unit = {
    assert(a.id === b.id)
    assert(a.predict === b.predict)
    assert(a.impurity === b.impurity)
    assert(a.isLeaf === b.isLeaf)
    assert(a.split === b.split)
    (a.stats, b.stats) match {
      case (Some(aStats), Some(bStats)) => assert(aStats.gain === bStats.gain)
      case (None, None) =>
      case _ => throw new AssertionError(
        s"Only one instance has stats defined. (a.stats: ${a.stats}, b.stats: ${b.stats})")
    }
    (a.leftNode, b.leftNode) match {
      case (Some(aNode), Some(bNode)) => checkTreeEqual(aNode, bNode)
      case (None, None) =>
      case _ => throw new AssertionError("Only one instance has leftNode defined. " +
          s"(a.leftNode: ${a.leftNode}, b.leftNode: ${b.leftNode})")
    }
    (a.rightNode, b.rightNode) match {
      case (Some(aNode: OldNode), Some(bNode: OldNode)) => checkTreeEqual(aNode, bNode)
      case (None, None) =>
      case _ => throw new AssertionError("Only one instance has rightNode defined. " +
          s"(a.rightNode: ${a.rightNode}, b.rightNode: ${b.rightNode})")
    }
  }

  /**
    * Check if the two trees are exactly the same.
    * If the trees are not equal, this prints the two trees and throws an exception.
    */
  private def checkModelEqual(a: DecisionTreeModel, b: DecisionTreeModel) = {
    try {
      assert(a.algo === b.algo)
      checkTreeEqual(a.topNode, b.topNode)
    } catch {
      case ex: Exception =>
        throw new AssertionError("checkEqual failed since the two trees were not " +
            "identical.\n" + "TREE A:\n" + a.toDebugString + "\n" +
            "TREE B:\n" + b.toDebugString + "\n", ex)
    }
  }

  test("model transform") {
    val gBTLRClassifier = new GBTLRClassifier()
    val model = gBTLRClassifier.fit(dataset)
    val prediction = model.transform(dataset)
    assert(model.lrModel.getFeaturesCol === "gbt_generated_features")
    val len1 = prediction.schema.fieldNames.length
    val len2 = dataset.schema.fieldNames.length
    assert(len1 === len2 + 4)
    assert(prediction.schema.fieldNames.contains("gbt_generated_features"))
  }
}

object  GBTLRClassifierSuite {

  def generateOrderedLabeledPoints(numFeatures: Int, numInstances: Int): Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](numInstances)
    for (i <- 0 until numInstances) {
      val label = if (i < numInstances / 10) {
        0.0
      } else if (i < numInstances / 2) {
        1.0
      } else if (i < numInstances * 0.9) {
        0.0
      } else {
        1.0
      }
      val features = Array.fill[Double](numFeatures)(i.toDouble)
      arr(i) = LabeledPoint(label, Vectors.dense(features))
    }
    arr
  }

  val allParamSettings: Map[String, Any] = Map(
    "seed" -> 123L,
    "subsamplingRate" -> 1.0,
    "GBTMaxIter" -> 20,
    "stepSize" -> 0.1,
    "maxDepth" -> 5,
    "maxBins" -> 32,
    "minInstancesPerNode" -> 1,
    "minInfoGain" -> 0.0,
    "checkpointInterval" -> 10,
    "fitIntercept" -> true,
    "probabilityCol" -> "probability",
    "rawPredictionCol" -> "rawPrediction",
    "standardization" -> true,
    "threshold" -> 0.5,
    "lossType" -> "logistic",
    "cacheNodeIds" -> false,
    "maxMemoryInMB" -> 256,
    "regParam" -> 0.0,
    "elasticNetParam" -> 0.0,
    "family" -> "auto",
    "LRMaxIter" -> 100,
    "tol" -> 1E-6,
    "aggregationDepth" -> 2
  )

}
