from __future__ import print_function
from pyspark.ml.classification import LogisticRegression                       
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import *
import pyspark.sql.functions as sf
import sys
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("heartdiseaseprediction").getOrCreate()

schema = StructType([
    StructField("male", IntegerType(), True),
    StructField("age", IntegerType(), True),
    StructField("education", IntegerType(), True),
    StructField("currentSmoker", IntegerType(), True),
    StructField("cigsPerDay", IntegerType(), True),
    StructField("BPMeds", IntegerType(), True),
    StructField("prevalentStroke", IntegerType(), True),
    StructField("prevalentHyp", IntegerType(), True),
    StructField("diabetes", IntegerType(), True),
    StructField("totChol", IntegerType(), True),
    StructField("sysBP", DoubleType(), True),
    StructField("diaBP", DoubleType(), True),
    StructField("BMI", DoubleType(), True),
    StructField("heartRate", DoubleType(), True),
    StructField("glucose", IntegerType(), True),
    StructField("TenYearCHD", IntegerType(), True)
])

heart_df = spark.read.csv("/heart-lr/input/framingham.csv", schema=schema, header=True)

heart_df.printSchema()

heart_df = heart_df.withColumnRenamed('male', 'Sex_male')

heart_df = heart_df.drop('education')

print("There are", heart_df.count(),
      "rows", len(heart_df.columns),
      "columns in the data.")
print("Here are the null values in the dataset, per column:")
heart_df.select([sf.count(sf.when(sf.isnull(c), c))\
          .alias(c) for c in heart_df.columns]).show()

print("Removing the null values now")
heart_df = heart_df.dropna()


print("There are", heart_df.count(),
      "rows", len(heart_df.columns),
      "columns in the data.")

assembler = VectorAssembler(inputCols=['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose'],
                            outputCol='features')
heart_df = assembler.transform(heart_df)

#Logistic Regression Model

label_col = "TenYearCHD"

(training_data, test_data) = heart_df.randomSplit([0.8, 0.2], seed=123)

lr = LogisticRegression(featuresCol='features', labelCol=label_col, maxIter=10)

lr_model = lr.fit(training_data)


#Evaluate Model
lr_predictions = lr_model.transform(test_data)

correct_predictions = lr_predictions.filter(lr_predictions[label_col] == lr_predictions['prediction']).count()
total_predictions = lr_predictions.count()
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy}")

evaluator = BinaryClassificationEvaluator(labelCol=label_col)
area_under_curve = evaluator.evaluate(lr_predictions)
print(f"Area under ROC curve: {area_under_curve}")
