import warnings
from datetime import datetime
import colorama
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import Window

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
            df_features = (
                spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("escape", "\\")
                .csv(ALL_DATASET_CSV_PATH)
            )
            df_features = Utilities.clean_up_df(df_features)

            df_features = df_features.withColumn(
                "Timestamp",
                F.coalesce(
                    F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss.SSSSSS"),
                    F.to_timestamp("Timestamp", "yyyy-MM-dd HH:mm:ss")
                )
            )

            df_features = df_features.orderBy(["Node_block_id", "Timestamp"])
            df_features = df_features.select(
                "Timestamp", "Date", "Time", "Content",
                "processed_EventTemplate", "Node_block_id", "Label"
            )

            # Get unique IDs
            unique_ids_df = df_features.select("Node_block_id").distinct()

            # -------------------------
            # Dataset-specific splitting
            # -------------------------
            if dataset == 'UU_HDFS':
                window = Window.orderBy("Node_block_id")
                unique_ids_df = unique_ids_df.withColumn("row_num", F.row_number().over(window))

                # Calculate limits purely in Spark
                total_ids_df = unique_ids_df.agg(F.max("row_num").alias("max_row"))
                total_ids_df = total_ids_df.withColumn("train_end", (0.6 * F.col("max_row")).cast("int")) \
                    .withColumn("val_end", (0.7 * F.col("max_row")).cast("int"))
                # Join limits back to assign splits
                unique_ids_df = unique_ids_df.crossJoin(total_ids_df)

                train_ids_df = unique_ids_df.filter(F.col("row_num") <= F.col("train_end")).select("Node_block_id")
                val_ids_df = unique_ids_df.filter((F.col("row_num") > F.col("train_end")) &
                                                  (F.col("row_num") <= F.col("val_end"))).select("Node_block_id")
                test_ids_df = unique_ids_df.filter(F.col("row_num") > F.col("val_end")).select("Node_block_id")

            elif dataset in ['UU_BGL', 'UU_TH']:
                unique_ids_df = unique_ids_df.withColumn("rand_val", F.rand())
                window = Window.orderBy(F.monotonically_increasing_id())
                unique_ids_df = unique_ids_df.withColumn("row_num", F.row_number().over(window))

                total_ids_df = unique_ids_df.agg(F.max("row_num").alias("max_row"))
                total_ids_df = total_ids_df.withColumn("train_end", (0.6 * F.col("max_row")).cast("int")) \
                    .withColumn("val_end", (0.7 * F.col("max_row")).cast("int"))
                unique_ids_df = unique_ids_df.crossJoin(total_ids_df)

                train_ids_df = unique_ids_df.filter(F.col("row_num") <= F.col("train_end")).select("Node_block_id")
                val_ids_df = unique_ids_df.filter((F.col("row_num") > F.col("train_end")) &
                                                  (F.col("row_num") <= F.col("val_end"))).select("Node_block_id")
                test_ids_df = unique_ids_df.filter(F.col("row_num") > F.col("val_end")).select("Node_block_id")
            else:
                raise ValueError(f"[ERROR] Unsupported dataset type: {dataset}")

            # -------------------------
            # Join splits with original dataset
            # -------------------------
            train_df = df_features.join(train_ids_df, on="Node_block_id", how="inner").withColumn("Type_ds",
                                                                                                  F.lit("Train"))
            val_df = df_features.join(val_ids_df, on="Node_block_id", how="inner").withColumn("Type_ds",
                                                                                              F.lit("Validation"))
            test_df = df_features.join(test_ids_df, on="Node_block_id", how="inner").withColumn("Type_ds",
                                                                                                F.lit("Test"))

            return train_df, val_df, test_df, df_features

    @staticmethod
    def processing_data_portion(train_df: DataFrame, validate_df: DataFrame, test_df: DataFrame, df_features: DataFrame, spark: SparkSession):
        """Create labeled and unlabeled portions for training and mark test set in PySpark without using collect or count."""

        # Get unique normal/anomaly Node_block_ids
        unique_normal_df = train_df.filter(F.col("Label") == "Normal").select("Node_block_id").distinct().cache()
        unique_anomaly_df = train_df.filter(F.col("Label") == "Anomaly").select("Node_block_id").distinct().cache()

        # Select ~50% of normal blocks for labeled training using sample
        selected_normal_50_df = unique_normal_df.sample(fraction=0.5, seed=42).cache()

        df_train_normal_50 = train_df.join(selected_normal_50_df, on="Node_block_id") \
            .withColumn("Temp_label", F.lit(0))

        # Remaining normal blocks (left_anti join) + all anomaly blocks are unlabeled
        remaining_normal_df = unique_normal_df.join(selected_normal_50_df, on="Node_block_id", how="left_anti")
        df_train_unlabeled = train_df.join(remaining_normal_df, on="Node_block_id", how="inner") \
            .union(train_df.join(unique_anomaly_df, on="Node_block_id", how="inner")) \
            .withColumn("Temp_label", F.lit(999))

        # Mark test set with Temp_label = 888
        test_df = test_df.withColumn("Temp_label", F.lit(888))

        # Combine all datasets
        final_dataset = df_train_normal_50.union(df_train_unlabeled).union(test_df).cache()

        final_dataset.printSchema()
        return final_dataset
