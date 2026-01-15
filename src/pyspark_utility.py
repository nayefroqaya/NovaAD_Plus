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

    def dataset_splitting(
        spark,
        pkl_path,
        dataset,
        round_id,
        Mix_or_stable
    ):
        """
        PKL (Pandas) → Spark → Split → Parquet
        """

        print(f"[INFO] Loading PKL dataset: {pkl_path}")

        # ==========================================================
        # 1. LOAD PKL USING PANDAS (ONLY PLACE WHERE PANDAS IS USED)
        # ==========================================================
        pdf = pd.read_pickle(pkl_path)

        # Optional cleanup if you had it before
        pdf = pdf.dropna().reset_index(drop=True)

        # ==========================================================
        # 2. CONVERT TO SPARK DATAFRAME
        # ==========================================================
        df_features = spark.createDataFrame(pdf)

        # ==========================================================
        # 3. TIMESTAMP STANDARDIZATION
        # ==========================================================
        df_features = (
            df_features
            .withColumn("Timestamp", F.to_timestamp("Timestamp"))
            .orderBy("Node_block_id", "Timestamp")
        )

        df_features = df_features.select(
            "Timestamp",
            "Date",
            "Time",
            "Content",
            "Original_Label",
            "EventId",
            "EventTemplate",
            "processed_EventTemplate",
            "Node_block_id",
            "Label"
        )

        print("[INFO] Timestamp standardized and sorted")

        # ==========================================================
        # 4. SPLIT BY Node_block_id (NO OVERLAP)
        # ==========================================================
        node_ids = (
            df_features
            .select("Node_block_id")
            .distinct()
            .rdd
            .map(lambda r: r[0])
            .collect()
        )

        np.random.shuffle(node_ids)

        total = len(node_ids)
        train_end = int(0.6 * total)
        val_end = train_end + int(0.1 * total)

        train_ids = node_ids[:train_end]
        val_ids   = node_ids[train_end:val_end]
        test_ids  = node_ids[val_end:]

        # ==========================================================
        # 5. CREATE SPLITS
        # ==========================================================
        train_df = (
            df_features
            .filter(F.col("Node_block_id").isin(train_ids))
            .withColumn("Type_ds", F.lit("Train"))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        val_df = (
            df_features
            .filter(F.col("Node_block_id").isin(val_ids))
            .withColumn("Type_ds", F.lit("Validation"))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        test_df = (
            df_features
            .filter(F.col("Node_block_id").isin(test_ids))
            .withColumn("Type_ds", F.lit("Test"))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        # Materialize cache
        train_df.count()
        val_df.count()
        test_df.count()

        # ==========================================================
        # 6. SAVE AS PARQUET (SPARK-NATIVE)
        # ==========================================================
        if dataset == "S_BGL" and Mix_or_stable == "0":
            suffix = "Stable"
        elif dataset == "S_BGL" and Mix_or_stable == "1":
            suffix = "Mix"
        else:
            suffix = "Default"

        save_path = f"../datasets/{dataset}/{round_id}_{dataset}_{suffix}_Splitted_Datasets"
        os.makedirs(save_path, exist_ok=True)

        train_df.write.mode("overwrite").parquet(os.path.join(save_path, "train_df"))
        val_df.write.mode("overwrite").parquet(os.path.join(save_path, "val_df"))
        test_df.write.mode("overwrite").parquet(os.path.join(save_path, "test_df"))

        print(f"[INFO] Saved splits to {save_path}")

        # ==========================================================
        # 7. STATS (BLOCK-LEVEL)
        # ==========================================================
        def block_stats(df, name):
            blocks = df.select("Node_block_id", "Label").dropDuplicates()
            normal = blocks.filter(F.col("Label") == "Normal").count()
            anomaly = blocks.filter(F.col("Label") == "Anomaly").count()
            print(f"{name} → Normal: {normal}, Anomaly: {anomaly}")

        block_stats(train_df, "Train")
        block_stats(test_df, "Test")

        print("[INFO] Dataset splitting complete")

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
