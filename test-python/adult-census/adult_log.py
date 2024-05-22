from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import sys

if __name__ == "__main__":
    spark = SparkSession.builder.appName("AdultCensusLogReg").getOrCreate()

    # Load data
    data = spark.read.csv(sys.argv[1], header=True, inferSchema=True)

    # Filter data to include only valid income labels
    data = data.filter((col("income") == " <=50K") | (col("income") == " >50K"))

    # Remove rows where any column contains 'NA' or '?'
    data = data.replace('?', None).na.drop()

    # List of categorical and numerical columns
    categoricalCols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    numericCols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # Indexing categorical columns
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(data) for column in categoricalCols]

    # Assemble features
    assemblerInputs = [c + "_index" for c in categoricalCols] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

    # Indexer for the 'income' column, ensuring binary output
    labelIndexer = StringIndexer(inputCol="income", outputCol="income_index").fit(data)

    # Logistic Regression Model
    lr = LogisticRegression(labelCol="income_index", featuresCol="features")

    # Pipeline
    pipeline = Pipeline(stages=indexers + [assembler, labelIndexer, lr])

    # Split data into training and test sets
    trainData, testData = data.randomSplit([0.8, 0.2], seed=1234)

    # Train model
    model = pipeline.fit(trainData)

    # Make predictions
    predictions = model.transform(testData)

    # Evaluate model using AUC
    binaryEvaluator = BinaryClassificationEvaluator(labelCol="income_index", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = binaryEvaluator.evaluate(predictions)
    print("Area Under ROC = %g" % auc)

    # Evaluate model using Accuracy
    accuracyEvaluator = MulticlassClassificationEvaluator(labelCol="income_index", predictionCol="prediction", metricName="accuracy")
    accuracy = accuracyEvaluator.evaluate(predictions)
    print("Test Accuracy = %g" % accuracy)

    spark.stop()
