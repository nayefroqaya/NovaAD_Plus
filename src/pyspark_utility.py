import colorama
import numpy as np
import warnings
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_timestamp
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import col, to_timestamp, lit
import pandas as pd

import os
import numpy as np
# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize colorama
colorama.init()
GREEN = colorama.Fore.GREEN
YELLOW = colorama.Fore.YELLOW
RESET = colorama.Fore.RESET
GRAY = colorama.Fore.LIGHTBLACK_EX


class Utilities:

    @staticmethod
    def insert_rows(df: DataFrame, row: dict, spark: SparkSession):
        """Insert a row at the end of the DataFrame using Spark union."""
        new_row_df = spark.createDataFrame([row], schema=df.schema)
        return df.union(new_row_df)

    @staticmethod
    def clean_up_df(df_features: DataFrame) -> DataFrame:
        """Remove rows with null content and display summary."""
        df_features = df_features.filter(F.col("Content").isNotNull())
        print(GREEN + "[INFO] DataFrame cleaned. Null 'Content' rows removed." + RESET)
        df_features.printSchema()
        print(f"[INFO] Total rows after cleanup: {df_features.count()}")
        return df_features

    @staticmethod
    def dataset_splitting(all_data_df, dataset, round, Mix_or_stable, spark):

        # =============================
        # Load dataset (already loaded)
        # =============================

        if Mix_or_stable == '0' and dataset == 'S_BGL':  # Stable
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = spark.read.option("header", True).csv(
                "../datasets/S_BGL/stable_equal_subset.csv"
            )

        elif Mix_or_stable == '1' and dataset == 'S_BGL':  # Mix
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = spark.read.option("header", True).csv(
                "../datasets/S_BGL/50_50_mixed_subset.csv"
            )

        else:
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = all_data_df

        df_features.printSchema()
        print(f"[INFO] Row count: {df_features.count()}")

        # =============================
        # Clean data
        # =============================
        df_features = Utilities.clean_up_df(df_features)

        # =============================
        # Timestamp standardization
        # =============================
        def update_timestamp(original_timestamp,
                             desired_format='%Y-%m-%d %H:%M:%S.%f'):
            try:
                datetime.strptime(str(original_timestamp), desired_format)
                return str(original_timestamp)
            except ValueError:
                parsed_timestamp = datetime.strptime(
                    str(original_timestamp), '%Y-%m-%d %H:%M:%S'
                )
                return parsed_timestamp.strftime(desired_format)

        update_timestamp_udf = F.udf(update_timestamp, StringType())

        df_features = df_features.withColumn(
            "Timestamp",
            update_timestamp_udf(F.col("Timestamp"))
        )

        df_features = df_features.orderBy(
            F.col("Node_block_id"),
            F.col("Timestamp")
        )

        df_features = df_features.select(
            'Timestamp', 'Date', 'Time', 'Content', 'Original_Label',
            'EventId', 'EventTemplate', 'processed_EventTemplate',
            'Node_block_id', 'Label'
        )

        df_features.printSchema()
        print(GREEN + "[INFO] Dataset timestamps standardized and sorted." + RESET)

        # =============================
        # Dataset splitting
        # =============================
        unique_ids_df = df_features.select("Node_block_id").distinct()
        total_ids = unique_ids_df.count()

        if dataset in ['HDFS', 'BGL', 'HDO', 'SP_100MB', 'SP_150MB',
                       'TH_1G', 'TH_2G', 'TH_5G', 'S_BGL']:

            shuffled_ids_df = unique_ids_df.orderBy(F.rand())

            train_size = int(0.6 * total_ids)
            val_size = int(0.1 * total_ids)

            train_ids = shuffled_ids_df.limit(train_size)
            val_ids = shuffled_ids_df.subtract(train_ids).limit(val_size)
            test_ids = shuffled_ids_df.subtract(train_ids).subtract(val_ids)

        else:
            raise ValueError(f"[ERROR] Unsupported dataset type: {dataset}")

        # =============================
        # Check for overlaps
        # =============================
        train_set = set(r[0] for r in train_ids.collect())
        val_set = set(r[0] for r in val_ids.collect())
        test_set = set(r[0] for r in test_ids.collect())

        print(YELLOW + f"[CHECK] Intersection train_val: {train_set & val_set}" + RESET)
        print(YELLOW + f"[CHECK] Intersection train_test: {train_set & test_set}" + RESET)
        print(YELLOW + f"[CHECK] Intersection val_test: {val_set & test_set}" + RESET)

        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError("[ERROR] Overlaps detected between dataset splits!")
        else:
            print(GREEN + "[INFO] No overlaps found between train, validation, and test sets." + RESET)

        # =============================
        # Create split DataFrames
        # =============================
        train_df = (
            df_features.join(train_ids, "Node_block_id", "inner")
            .withColumn("Type_ds", F.lit("Train"))
        )

        val_df = (
            df_features.join(val_ids, "Node_block_id", "inner")
            .withColumn("Type_ds", F.lit("Validation"))
        )

        test_df = (
            df_features.join(test_ids, "Node_block_id", "inner")
            .withColumn("Type_ds", F.lit("Test"))
        )

        # =============================
        # Save datasets
        # =============================
        if Mix_or_stable == '0' and dataset == 'S_BGL':
            save_path = f"../datasets/{dataset}/{round}_{dataset}_Stable_Splitted_Datasets"

        elif Mix_or_stable == '1' and dataset == 'S_BGL':
            save_path = f"../datasets/{dataset}/{round}_{dataset}_Mix_Splitted_Datasets"

        else:
            save_path = f"../datasets/{dataset}/{round}_{dataset}_Splitted_Datasets"

        os.makedirs(save_path, exist_ok=True)

        train_df.write.mode("overwrite").parquet(os.path.join(save_path, "train_df"))
        val_df.write.mode("overwrite").parquet(os.path.join(save_path, "val_df"))
        test_df.write.mode("overwrite").parquet(os.path.join(save_path, "test_df"))


        # =============================
        # Display split info
        # =============================
        print(
            GREEN +
            f"[INFO] Dataset split complete. Sizes -> "
            f"Train: {train_df.count()}, "
            f"Validation: {val_df.count()}, "
            f"Test: {test_df.count()}" +
            RESET
        )

        # =============================
        # Block-level statistics
        # =============================
        df_block_train = train_df.dropDuplicates(['Node_block_id'])
        df3 = df_block_train.filter(F.col("Label") == "Normal")
        df4 = df_block_train.filter(F.col("Label") == "Anomaly")

        print(' Normal seq Train : ' + str(df3.count()))
        print(' Anomaly seq Train : ' + str(df4.count()))

        df_block_test = test_df.dropDuplicates(['Node_block_id'])
        df3 = df_block_test.filter(F.col("Label") == "Normal")
        df4 = df_block_test.filter(F.col("Label") == "Anomaly")

        print(' Normal seq Test : ' + str(df3.count()))
        print(' Anomaly seq Test : ' + str(df4.count()))

        return train_df, val_df, test_df, df_features


    @staticmethod
    def processing_data_portion(train_df: DataFrame, validate_df: DataFrame, test_df: DataFrame):
        """Create labeled and unlabeled portions for training and mark test set in PySpark without using collect or count."""

        # 1️⃣ Separate normal and anomaly Node_block_id
        unique_ids_df = train_df.select("Node_block_id", "Label").distinct().cache()

        unique_normal_df = unique_ids_df.filter(F.col("Label") == "Normal")
        unique_anomaly_df = unique_ids_df.filter(F.col("Label") == "Anomaly")

        # Count
        count_normal = unique_normal_df.count()
        count_anomaly = unique_anomaly_df.count()

        print(f"[INFO] Total unique normal Node_block_ids: {count_normal}")
        print(f"[INFO] Total unique anomaly Node_block_ids: {count_anomaly}")

        # 2️⃣ Assign row numbers WITHOUT shuffling
        w = Window.orderBy("Node_block_id")
        # deterministic ordering instead of random

        normal_ordered = unique_normal_df.withColumn("row_num", F.row_number().over(w))

        # Select 50% for labeling
        split_point = count_normal // 2

        normal_labeled_ids_part1_50 = normal_ordered.filter(F.col("row_num") <= split_point).select("Node_block_id")

        normal_unlabeled_ids_part2_50 = normal_ordered.filter(F.col("row_num") > split_point).select("Node_block_id")

        # 3️⃣ Get all anomaly IDs as unlabeled
        anomaly_ids = unique_anomaly_df.select("Node_block_id")

        # 4️⃣ Create labeled dataset (Temp_label = 0)
        df_train_normal_50 = train_df.join(normal_labeled_ids_part1_50, on="Node_block_id", how="inner").withColumn(
            "Temp_label", F.lit(0))

        # 5️⃣ Create unlabeled dataset (Temp_label = 999)
        remaining_unlabeled_ids = normal_unlabeled_ids_part2_50.union(anomaly_ids).distinct()

        df_train_unlabeled = train_df.join(remaining_unlabeled_ids, on="Node_block_id", how="inner").withColumn(
            "Temp_label", F.lit(999))

        # 6️⃣ Mark test set (Temp_label = 888)
        df_test_labeled = test_df.withColumn("Temp_label", F.lit(888))

        validate_df_labeled = validate_df.withColumn("Temp_label", F.lit(777))


        # 7️⃣ Combine all
        final_dataset = df_train_normal_50.unionByName(df_train_unlabeled).unionByName(df_test_labeled).unionByName(validate_df_labeled)
        final_dataset.printSchema()
        return final_dataset
