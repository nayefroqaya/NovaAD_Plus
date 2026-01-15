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

    def dataset_splitting(All_dataset_path_as_csv, dataset, round, Mix_or_stable, spark):

        print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)

        # -------------------------------------------------
        # Load dataset (CSV path OR Spark DataFrame)
        # -------------------------------------------------
        if isinstance(All_dataset_path_as_csv, DataFrame):
            df_features = All_dataset_path_as_csv
        else:
            df_features = (
                spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("escape", "\\")
                .csv(All_dataset_path_as_csv)
            )

        df_features = df_features.cache()
        df_features.count()

        # -------------------------------------------------
        # Clean data (Spark version required)
        # -------------------------------------------------
        df_features = Utilities.clean_up_df(df_features)

        # -------------------------------------------------
        # Standardize timestamps
        # -------------------------------------------------
        df_features = df_features.withColumn(
            "Timestamp",
            to_timestamp(col("Timestamp"), "yyyy-MM-dd HH:mm:ss.SSS")
        )

        # -------------------------------------------------
        # Sort and select columns
        # -------------------------------------------------
        df_features = (
            df_features
            .orderBy("Node_block_id", "Timestamp")
            .select(
                "Timestamp", "Date", "Time", "Content", "Original_Label",
                "EventId", "EventTemplate", "processed_EventTemplate",
                "Node_block_id", "Label"
            )
        )

        print(GREEN + "[INFO] Dataset timestamps standardized and sorted." + RESET)

        # -------------------------------------------------
        # Train / Validation / Test split by Node_block_id
        # -------------------------------------------------
        unique_ids = (
            df_features
            .select("Node_block_id")
            .distinct()
            .rdd
            .map(lambda x: x[0])
            .collect()
        )

        total_ids = len(unique_ids)

        if dataset in ['HDFS', 'BGL', 'HDO', 'SP_100MB', 'SP_150MB',
                       'TH_1G', 'TH_2G', 'S_BGL']:

            shuffled_ids = np.random.permutation(unique_ids)
            train_size = int(0.6 * total_ids)
            val_size = int(0.1 * total_ids)

            train_ids = shuffled_ids[:train_size]
            val_ids = shuffled_ids[train_size:train_size + val_size]
            test_ids = shuffled_ids[train_size + val_size:]

        else:
            raise ValueError(f"[ERROR] Unsupported dataset type: {dataset}")

        # -------------------------------------------------
        # Create split DataFrames (FIXED string column)
        # -------------------------------------------------
        train_df = df_features.filter(col("Node_block_id").isin(list(train_ids)))
        train_df = train_df.withColumn("Type_ds", lit("Train"))

        val_df = df_features.filter(col("Node_block_id").isin(list(val_ids)))
        val_df = val_df.withColumn("Type_ds", lit("Validation"))

        test_df = df_features.filter(col("Node_block_id").isin(list(test_ids)))
        test_df = test_df.withColumn("Type_ds", lit("Test"))

        # -------------------------------------------------
        # Persist splits
        # -------------------------------------------------
        train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
        val_df = val_df.persist(StorageLevel.MEMORY_AND_DISK)
        test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

        train_count = train_df.count()
        val_count = val_df.count()
        test_count = test_df.count()

        print(
            GREEN +
            f"[INFO] Dataset split complete. Sizes -> "
            f"Train: {train_count}, Validation: {val_count}, Test: {test_count}"
            + RESET
        )

        # -------------------------------------------------
        # Block-level statistics (Spark equivalent)
        # -------------------------------------------------
        def print_stats(df, name):
            blocks = df.select("Node_block_id", "Label").distinct()
            normal = blocks.filter(col("Label") == "Normal").count()
            anomaly = blocks.filter(col("Label") == "Anomaly").count()
            print(f" Normal seq {name} : {normal}")
            print(f" Anomaly seq {name} : {anomaly}")

        print_stats(train_df, "Train")
        print_stats(test_df, "Test")

        return train_df, val_df, test_df, df_features




    @staticmethod
    def processing_data_portion(train_df: DataFrame, validate_df: DataFrame, test_df: DataFrame, df_features: DataFrame,
                                spark: SparkSession):
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

        # 7️⃣ Combine all
        final_dataset = df_train_normal_50.unionByName(df_train_unlabeled).unionByName(df_test_labeled)

        final_dataset.printSchema()
        return final_dataset
