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
        # Load dataset
        # =============================
        if Mix_or_stable == '0' and dataset == 'S_BGL':  # Stable
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = spark.read.option("header", True).csv("../datasets/S_BGL/stable_equal_subset.csv")

        elif Mix_or_stable == '1' and dataset == 'S_BGL':  # Mix
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = spark.read.option("header", True).csv("../datasets/S_BGL/50_50_mixed_subset.csv")

        else:
            print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
            df_features = all_data_df

        # =============================
        # Clean data
        # =============================
        df_features = Utilities.clean_up_df(df_features)

        # =============================
        # Timestamp parsing (Spark-native, handles with/without microseconds)
        # =============================
        ts_micro = F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss.SSSSSS")
        ts_sec   = F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss")
        df_features = df_features.withColumn("Timestamp_ts", F.coalesce(ts_micro, ts_sec))

        bad_ts = df_features.filter(F.col("Timestamp_ts").isNull()).count()
        if bad_ts > 0:
            print(YELLOW + f"[WARN] {bad_ts} rows have unparsed Timestamp_ts (NULL)." + RESET)

        # Keep only the columns you want (keep Timestamp_ts for ordering/splitting)
        df_features = df_features.select(
            'Timestamp', 'Timestamp_ts', 'Date', 'Time', 'Content', 'Original_Label',
            'EventId', 'EventTemplate', 'processed_EventTemplate',
            'Node_block_id', 'Label'
        )

        # Order logs inside each block by time (sequence correctness)
        df_features = df_features.orderBy(F.col("Node_block_id"), F.col("Timestamp_ts"))

        print(GREEN + "[INFO] Dataset timestamps parsed and sorted." + RESET)

        # =============================
        # Chronological split by Node_block_id
        # =============================
        #  dataset == 'TH_2G_ratio' or dataset == 'TH_5G_ratio'
        supported = {'HDFS','BGL','HDO','SP_100MB','SP_150MB','SP_100MB_ratio','SP_150MB_ratio','TH_1G', 'TH_1G_ratio','TH_2G','TH_2G_ratio', 'TH_3G_ratio','TH_5G','TH_5G_ratio', 'TH_6G_ratio','TH_9G_ratio','S_BGL'}
        if dataset not in supported:
            raise ValueError(f"[ERROR] Unsupported dataset type: {dataset}")
        # One row per block with its start time (sequence time)
        block_time_df = (
            df_features
            .groupBy("Node_block_id")
            .agg(F.min("Timestamp_ts").alias("block_start_ts"))
        )

        # Deterministic chronological ordering (tie-break by id)
        w = Window.orderBy(F.col("block_start_ts").asc(), F.col("Node_block_id").asc())
        ordered_blocks = block_time_df.withColumn("rn", F.row_number().over(w)).cache()
        total_blocks = ordered_blocks.count()  # materialize

        train_size = int(0.6 * total_blocks)
        val_size   = int(0.1 * total_blocks)

        train_ids = ordered_blocks.filter(F.col("rn") <= train_size).select("Node_block_id").cache()
        val_ids   = ordered_blocks.filter((F.col("rn") > train_size) & (F.col("rn") <= train_size + val_size)).select("Node_block_id").cache()
        test_ids  = ordered_blocks.filter(F.col("rn") > train_size + val_size).select("Node_block_id").cache()

        _ = train_ids.count(); _ = val_ids.count(); _ = test_ids.count()

        # =============================
        # Overlap checks (no collect)
        # =============================
        if (train_ids.join(val_ids, "Node_block_id").limit(1).count() > 0 or
            train_ids.join(test_ids, "Node_block_id").limit(1).count() > 0 or
            val_ids.join(test_ids, "Node_block_id").limit(1).count() > 0):
            raise ValueError("[ERROR] Overlaps detected between dataset splits!")
        else:
            print(GREEN + "[INFO] No overlaps found between train, validation, and test sets." + RESET)

        # Optional: verify chronological guarantee
        max_train_start = ordered_blocks.filter(F.col("rn") <= train_size).agg(F.max("block_start_ts")).first()[0]
        min_test_start  = ordered_blocks.filter(F.col("rn") > train_size + val_size).agg(F.min("block_start_ts")).first()[0]
        print(YELLOW + f"[CHECK] max(train block start)={max_train_start}, min(test block start)={min_test_start}" + RESET)

        # =============================
        # Create split DataFrames (and keep ordered)
        # =============================
        train_df = (
            df_features.join(train_ids, "Node_block_id", "inner")
            .withColumn("Type_ds", F.lit("Train"))
            .orderBy("Node_block_id", "Timestamp_ts")
        )

        val_df = (
            df_features.join(val_ids, "Node_block_id", "inner")
            .withColumn("Type_ds", F.lit("Validation"))
            .orderBy("Node_block_id", "Timestamp_ts")
        )

        test_df = (
            df_features.join(test_ids, "Node_block_id", "inner")
            .withColumn("Type_ds", F.lit("Test"))
            .orderBy("Node_block_id", "Timestamp_ts")
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
        print(' Normal seq Train : ' + str(df_block_train.filter(F.col("Label") == "Normal").count()))
        print(' Anomaly seq Train : ' + str(df_block_train.filter(F.col("Label") == "Anomaly").count()))

        df_block_test = test_df.dropDuplicates(['Node_block_id'])
        print(' Normal seq Test : ' + str(df_block_test.filter(F.col("Label") == "Normal").count()))
        print(' Anomaly seq Test : ' + str(df_block_test.filter(F.col("Label") == "Anomaly").count()))


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
