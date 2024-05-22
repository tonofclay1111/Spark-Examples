import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType



spark = (SparkSession.builder
                  .appName('Toxic Comment Classification')
                  .enableHiveSupport()
                  .config("spark.executor.memory", "4G")
                  .config("spark.driver.memory","18G")
                  .config("spark.executor.cores","7")
                  .config("spark.python.worker.memory","4G")
                  .config("spark.driver.maxResultSize","0")
                  .config("spark.sql.crossJoin.enabled", "true")
                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                  .config("spark.default.parallelism","2")
                  .getOrCreate())

spark.sparkContext.setLogLevel('INFO')

def to_spark_df(fin):
    df = pd.read_csv(fin)
    df.fillna("", inplace=True)
    return spark.createDataFrame(df)


train_path = "/spark-examples/test-data/toxic_class_train.csv"
test_path = "/spark-examples/test-data/toxic_class_test.csv"

train_df = to_spark_df(train_path)
test_df = to_spark_df(test_path)


# Basic sentence tokenizer: Splits the text into a list of words
tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
# Apply the tokenizer to transform the dataset
words_data = tokenizer.transform(train_df)

# Count the words in a document: Converts words to numerical feature vectors
hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures")
# Transform the words dataset to feature vectors
tf = hashing_tf.transform(words_data)

# Build the IDF model and transform the original token frequencies into their TF-IDF counterparts
idf = IDF(inputCol="rawFeatures", outputCol="features")
# Fit the IDF model to the feature vectors
idf_model = idf.fit(tf)
# Apply the IDF model to the feature vectors to scale them appropriately
tfidf = idf_model.transform(tf)

# Regularization parameter
REG = 0.1

# Define the Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)

# Fit the model to a limited subset of data for performance considerations
lrModel = lr.fit(tfidf.limit(5000))

# Transform the training data to get predictions
res_train = lrModel.transform(tfidf)

# Define a UDF to extract the probability of the positive class
extract_prob = udf(lambda x: float(x[1]), FloatType())

# Add a probability column and show the results
(res_train.withColumn("proba", extract_prob("probability"))
 .select("proba", "prediction")
 .show())

# Process the test data
test_tokens = tokenizer.transform(test_df)
test_tf = hashing_tf.transform(test_tokens)
test_tfidf = idf_model.transform(test_tf)

# Initialize an empty DataFrame to store test results
test_res = test_df.select('id')

# Iterate through each label column to train and predict using the respective label
out_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']  # specify the label columns

for col in out_cols:
    print(f"Processing {col}")
    lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
    print("...fitting")
    lrModel = lr.fit(tfidf)  # Note: Fit the model to the entire training data or a representative subset
    print("...predicting")
    res = lrModel.transform(test_tfidf)
    print("...appending result")
    test_res = test_res.join(res.select('id', 'probability'), on="id")
    print("...extracting probability")
    test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")

# Show the first few results
test_res.show(15)
