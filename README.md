# Spark-Examples
This repository contains our final project for Big Data Computing, focusing on utilizing Apache Spark on Google Cloud Platform for big data machine learning tasks.. The project encompasses several parts, each designed to enhance understanding and application of machine learning algorithms in a distributed computing environment.
## Background on Apache Spark
Apache Spark is an open-source, distributed computing system that provides an interface for programming entire clusters with implicit data parallelism and fault tolerance. It is designed to process large-scale data quickly and efficiently, making it ideal for big data analytics and machine learning tasks.
## How Apache Spark Works  
Spark operates on the concept of Resilient Distributed Datasets (RDDs), which are immutable collections of objects that can be processed in parallel across a cluster of computers. RDDs can be created from simple text files, SQL databases, NoSQL stores like HBase, or existing RDDs. Spark allows for in-memory processing, meaning that it can cache intermediate data in memory, reducing the need for repeated data retrieval from disk and significantly speeding up the computation.

Spark's architecture is based on a master-slave model, where a central driver program (master) coordinates the execution of tasks across a cluster of worker nodes (slaves). These tasks are distributed across the worker nodes, which process the data in parallel and return the results to the driver program.

## Benefits of Apache Spark 
1. Speed: Spark's in-memory processing capabilities make it significantly faster than traditional disk-based processing frameworks like Hadoop MapReduce. It can process data up to 100 times faster in memory and 10 times faster on disk.
2. Ease of Use: Spark provides high-level APIs in Java, Scala, Python, and R, making it accessible to a wide range of developers. Its interactive shell also allows for quick prototyping and iterative development.
3. Unified Engine: Spark offers a unified engine that supports a wide array of big data processing tasks, including batch processing, streaming, machine learning, and graph processing. This versatility makes it a one-stop solution for various data analytics needs.
4. Fault Tolerance: Spark's RDDs are designed to be fault-tolerant. If any partition of an RDD is lost, it can be recomputed using the lineage information stored in Spark, ensuring data reliability and consistency.
5. Scalability: Spark can scale up from a single machine to thousands of cluster nodes, making it suitable for both small-scale and large-scale data processing tasks.
6. Advanced Analytics: Spark includes advanced libraries for machine learning (MLlib), graph processing (GraphX), and stream processing (Structured Streaming), enabling complex analytics and real-time data processing.

With these advantages, Apache Spark has become a popular choice for organizations looking to harness the power of big data for actionable insights and data-driven decision-making.

## Accessing Spark on Google Cloud Platform
To leverage the power of Spark for our project, we utilized Google Cloud Platform (GCP) to set up and manage our Spark cluster. GCP's Dataproc service allows for quick and easy deployment of Spark clusters, providing a scalable and reliable environment for big data processing. By using GCP, we ensured that our Spark jobs were executed efficiently across a distributed infrastructure, facilitating the handling of large datasets and complex computations.

# 1: Toxic Comment Classification
Our goal here was to build a logistic regression model to classify internet comments as being toxic or non-toxic. Using the Jigsaw Toxic Comment Classification Challenge dataset, we developed a model that can accurately identify and predict toxic comments based on their text content.
## Download Data  
1. Start 3 node cluster from Google Cloud account and connect from command line using the external IP of the Manager node (instance-1). Then use su root command and provide password.
2. Navigate to the data folder ``` cd spark-examples/test-data ```
3. Install Kaggle API in the cluster environment: ``` pip install kaggle ```
4. Get your kaggle API token. Go to Kaggle.com, log in to your account. Navigate to your account settings (click on your profile picture, then "Account"). Scroll down to the "API" section and click on "Create New API Token".
   This will download a kaggle.json file containing your API credentials.
5. Upload Kaggle.json file to the 3 node cluster. I saved my json file to github, then used wget to download the file directly Save it in the spark cluster as ```~/.kaggle/kaggle.json``` Change permissions to ensure token is secure: ```chmod 600 ~/.kaggle/kaggle.json```
6. Navigate to /spark-examples/test-data to download data via the following command ```kaggle competitions download -c jigsaw-toxic-comment-classification-challenge```
7. Unzip files: These files came in zipped folders, so to access the csvs we need to unzip them. ```unzip jigsaw-toxic-comment-classification-challenge.zip``` ```unzip test.csv.zip``` ```unzip train.csv.zip```
8. Rename files to the following:
```toxic_class_test.csv```
```toxic_class_train.csv```
 
## **Using Spark Logistic Regression to classify toxic coments**
1. Within the folder spark-examples, run the following command: `bash start.sh`. This commands starts the spark instance
2. Navigate to spark-examples/test-python/toxic-comment
3. To run the full program, run the command ```spark-submit toxic-comment-classification.py```and confirm the output for the first 15 toxicity scores
### Example Output:
![image](https://github.com/tonofclay1111/Spark-Examples/assets/164271616/9f343650-1c1e-4250-a4e3-0ead7d99d1cd)

The results contain the probabilities for different toxicity labels.  The example output shows the first 15 comments. Inputs with probability values closer to 1 indicate comments that are more likely to be toxic. For example, the very first input in the screenshot has a score in the toxic column of .91. In this case, this comment would be flagged. Which would enable moderators to review the comment and facilitate the crackdown of harassment on the website.  



# 2: Heart Disease Prediction
Our objective was to use logistic regression to predict the risk of heart disease. By analyzing the Framingham Heart Study dataset, we wanted to identify the most significant risk factors and predict the likelihood of heart disease based on these factors.
## **To run program**
1. Ensure that dataset is saved in /spark-examples/test-data folder and named 'framingham.csv'
2. Navigate to /spark-examples/test-python/heart-lr folder
3. Ensure that there are 2 files there: heartlr.py and test.sh
4. Run the code by executing the command ```bash test.sh```
### Example Output:
![image](https://github.com/tonofclay1111/Spark-Examples/assets/164271616/b4843acd-32b8-431f-8ad3-68dc72772566)

When running test.sh, the top rows of the dataset (heart_df.show()) are visible so we can confirm that the data is coming through and proceed with the rest of the code. 

Our overall accuracy score was 87% and ROC was 72%. These scores are similar to the Kaggle project scores, and indicate that our model is effective in predicting heart disease. The high accuracy suggests that our model correctly classifies a significant proportion of instances, while the ROC score demonstrates that our model has a good balance between sensitivity and specificity. This indicates that our approach to data preprocessing, feature selection, and model tuning for logistic regression is robust and reliable for predicting heart disease.

# 3: Logistic Regression Classifier on Census Income Data
Our goal of this part was to create a logistic regression model, random forest model, and decision tree model to predict income levels based on demographic data. Using the Census Dataset from the UCI Machine Learning Repository, we trained a binary classifier to predict whether an individual's income exceeds $50K or not.
## **To run program**
1. Navigate to the data folder ``` cd spark-examples/test-data/ ``` and ensure adult_data.csv is correctly uploaded
2. Navigate to the python folder ``` cd spark-examples/test-python/adult-census ``` and ensure adult_log.py and test.sh files exist
3. To run the full program, run bash test.sh and confirm the output.
### Results:
1. Decision Tree: Accuracy = 84.01%; ROC = 68.61%
2. Logistic Regression: Accuracy = 83.3%; ROC = 88.07%
3. Random Forest: 83.97%; ROC = 89.62%

The different models have varying strengths and weaknesses in predicting income levels based on the Census data. The Random Forest model provides the best balance of accuracy and ROC, making it the most robust model for predicting income levels in this dataset. The high ROC scores for Logistic Regression and Random Forest also suggest these models are better at distinguishing between income classes than the Decision Tree model.
