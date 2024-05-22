#!/bin/bash
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /adult-census/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /adult-census/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../test-data/adult_data.csv /adult-census/input/
/usr/local/spark/bin/spark-submit --master spark://$SPARK_MASTER:7077 ./adult_log.py hdfs://$SPARK_MASTER:9000/adult-census/input/