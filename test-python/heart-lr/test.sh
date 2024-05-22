#!/bin/bash
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /heart-lr/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /heart-lr/output
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /heart-lr/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../test-data/framingham.csv /heart-lr/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./heartlr.py hdfs://$SPARK_MASTER:9000/heart-lr/input/
