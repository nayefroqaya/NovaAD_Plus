import colorama
import numpy as np
import warnings
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

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
    def dataset_splitting(ALL_DATASET_CSV_PATH: str, dataset: str, spark: SparkSession):

        print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
        df_features = (spark.read.option("header", True).option("inferSchema", True).option("escape", "\\").csv(
            ALL_DATASET_CSV_PATH))
        df_features = Utilities.clean_up_df(df_features)

        df_features = df_features.withColumn("Timestamp",
            F.coalesce(F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss.SSSSSS"),
                F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss")))

        df_features = df_features.orderBy(["Node_block_id", "Timestamp"])
        df_features = df_features.select("Timestamp", "Date", "Time", "Content", "processed_EventTemplate",
            "Node_block_id", "Label")

        # Get unique IDs
        unique_ids_df = df_features.select("Node_block_id").distinct()

        # -------------------------
        # Dataset-specific splitting
        # -------------------------
        if dataset == 'HDFS':

            # 2️⃣ Shuffle using random value
            unique_ids_df = unique_ids_df.withColumn("rand_val", F.rand())

            # 3️⃣ Assign row number after shuffling
            window = Window.orderBy("rand_val")
            unique_ids_df = unique_ids_df.withColumn("row_num", F.row_number().over(window))

            # 4️⃣ Compute split sizes
            total_ids = unique_ids_df.count()

            train_size = int(0.6 * total_ids)
            val_size = int(0.1 * total_ids)

            train_end = train_size
            val_end = train_size + val_size

            # 5️⃣ Create splits using row_num ranges
            train_ids_df = unique_ids_df.filter(F.col("row_num") <= train_end).select("Node_block_id")

            val_ids_df = unique_ids_df.filter((F.col("row_num") > train_end) & (F.col("row_num") <= val_end)).select(
                "Node_block_id")

            test_ids_df = unique_ids_df.filter(F.col("row_num") > val_end).select("Node_block_id")



        elif dataset in ['BGL', 'TH']:
            # 2️⃣ Shuffle using random value
            unique_ids_df = unique_ids_df.withColumn("rand_val", F.rand())

            # 3️⃣ Assign row number after shuffling
            window = Window.orderBy("rand_val")
            unique_ids_df = unique_ids_df.withColumn("row_num", F.row_number().over(window))

            # 4️⃣ Compute split sizes
            total_ids = unique_ids_df.count()

            train_size = int(0.6 * total_ids)
            val_size = int(0.1 * total_ids)

            train_end = train_size
            val_end = train_size + val_size

            # 5️⃣ Create splits using row_num ranges
            train_ids_df = unique_ids_df.filter(F.col("row_num") <= train_end).select("Node_block_id")

            val_ids_df = unique_ids_df.filter((F.col("row_num") > train_end) & (F.col("row_num") <= val_end)).select(
                "Node_block_id")

            test_ids_df = unique_ids_df.filter(F.col("row_num") > val_end).select("Node_block_id")


        else:
            raise ValueError(f"[ERROR] Unsupported dataset type: {dataset}")

        # -------------------------
        # Join splits with original dataset
        # -------------------------
        train_df = df_features.join(train_ids_df, on="Node_block_id", how="inner").withColumn("Type_ds", F.lit("Train"))
        val_df = df_features.join(val_ids_df, on="Node_block_id", how="inner").withColumn("Type_ds",
                                                                                          F.lit("Validation"))
        test_df = df_features.join(test_ids_df, on="Node_block_id", how="inner").withColumn("Type_ds", F.lit("Test"))

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
