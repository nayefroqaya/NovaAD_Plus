import colorama
import numpy as np
import os
import warnings
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark_anomaly_detection import AnomalyDetector
from pyspark_features_engineering import FeaturesEngineering
from pyspark_features_extracting import FeaturesExtractor
from pyspark_model_evaluation import ModelEvaluation
from pyspark_utility import Utilities
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import col
import psutil
import pandas as pd

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = str

# ---------------------------
# Initialize Spark Session
# ---------------------------
'''
spark = (SparkSession.builder.appName("SentimentAnalysisPySpark")  # Memory tuning
         .config("spark.executor.memory", "32g").config("spark.driver.memory", "32g").config(
    "spark.executor.memoryOverhead", "6g").config("spark.driver.memoryOverhead", "6g")

         # Use all CPU cores
         .config("spark.executor.cores", "8").config("spark.driver.cores", "8")

         # Use Arrow for speed
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")

         # ===============================
         # ENABLE ADAPTIVE QUERY EXECUTION
         # ===============================
         .config("spark.sql.adaptive.enabled", "true")  # Auto-optimize shuffle partitions
         .config("spark.sql.adaptive.shuffle.enabled", "true")  # Auto-merge small partitions
         .config("spark.sql.adaptive.coalescePartitions.enabled",
                 "true")  # This controls auto partition size (64 MB default)
         .config("spark.sql.adaptive.advisoryPartitionSizeInBytese", "64m")  # Set upper bound on partitions
         .config("spark.sql.adaptive.coalescePartitions.minPartitionSizem", "64MB")

         # Avoid huge default 200 partitions
         .config("spark.sql.shuffle.partitions", "200")

         .getOrCreate())

print(f"[INFO] Spark initialized with {spark.sparkContext.defaultParallelism} parallel tasks")
'''

# -----------------------------
# Determine free RAM safely
# -----------------------------
available_gb = psutil.virtual_memory().available / (1024 ** 3)
spark_memory_gb = max(4, int(available_gb * 0.8))  # use 80% of available RAM, at least 4 GB

print(f"Setting Spark memory to {spark_memory_gb} GB")

# -----------------------------
# Initialize Spark
# -----------------------------
spark = (
    SparkSession.builder
    .appName("SentimentAnalysisPySpark")
    .master("local[*]")  # use all CPU cores
    .config("spark.driver.memory", f"{spark_memory_gb}g")
    .config("spark.executor.memory", f"{spark_memory_gb}g")
    .config("spark.memory.fraction", "0.6")
    .config("spark.memory.storageFraction", "0.3")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

# ===================== ======================
warnings.filterwarnings('ignore')
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW




def main():

    # ---------------- Device setup ----------------
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.backends.cudnn.enabled = True

    # ---------------- Display options ----------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # ---------------- Project configuration ----------------
    DATASET = 'SP_150MB'
    DATASETS_FOLDER = 'datasets'
    Round = '1'
    mode = 'M'
    Mix_or_stable = '0'

    # ---------------- Spark session ----------------
    spark = SparkSession.builder \
        .appName("LogAnomalyPipeline") \
        .getOrCreate()



    # ---------------- Paths ----------------
    DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_doc_topic_df.pkl'
    SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_sentiment_df.pkl'
    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_pre_final_global_features.pkl'

    # ---------------- Initialize classes ----------------
    logdata_read_obj = LogdataRead()
    features_extracting_obj = FeaturesExtractor()
    features_engineering_obj = FeaturesEngineering()
    anomaly_detection_obj = AnomalyDetector()
    model_evaluation_obj = ModelEvaluation()
    utilities_obj = Utilities()

    # ---------------- Process normal data ----------------
    print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")

    if Mix_or_stable == '0' and DATASET == 'S_BGL':
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_Stable_Splitted_Datasets")
    elif Mix_or_stable == '1' and DATASET == 'S_BGL':
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_Mix_Splitted_Datasets")
    else:
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_Splitted_Datasets")

    # ---------------- Load PKL splits (Pandas → Spark) ----------------
    train_df = spark.createDataFrame(
        pd.read_pickle(os.path.join(save_path, "train_df.pkl"))
    ).cache()

    val_df = spark.createDataFrame(
        pd.read_pickle(os.path.join(save_path, "val_df.pkl"))
    ).cache()

    test_df = spark.createDataFrame(
        pd.read_pickle(os.path.join(save_path, "test_df.pkl"))
    ).cache()

    train_df.count()
    val_df.count()
    test_df.count()
    exit()

    # ---------------- Merge datasets ----------------
    final_train_with_test_with_val = utilities_obj.processing_data_portion(
        train_df, val_df, test_df, spark
    ).persist(StorageLevel.MEMORY_AND_DISK)

    final_train_with_test_with_val.count()

    # ---------------- Features Extracting ----------------
    print(f"{GRAY}Extracting features for training and test datasets...{RESET}")

    number_component, best_topic_number = \
        features_extracting_obj.features_extracting_configuring_tuning(
            features_extracting_obj,
            DOC_TOPIC_DF_PATH,
            SENTIMENT_DF_PATH,
            DATASET,
            PRE_FINAL_GLOBAL_FEATURES_PKL_PATH,
            final_train_with_test_with_val
        )

    # ---------------- Load feature PKL → Spark ----------------
    final_train_with_test_with_val = spark.createDataFrame(
        pd.read_pickle(PRE_FINAL_GLOBAL_FEATURES_PKL_PATH)
    ).persist(StorageLevel.MEMORY_AND_DISK)

    final_train_with_test_with_val.count()

    # ---------------- Features Engineering ----------------
    print(f"{GRAY}Aggregating and transforming features...{RESET}")

    sequences_df, x_sequences_df, y_sequences_df = \
        features_engineering_obj.features_aggregation_transformation(
            final_train_with_test_with_val,
            DATASET
        )

    sequences_df = sequences_df.persist(StorageLevel.MEMORY_AND_DISK)
    x_sequences_df = x_sequences_df.cache()

    sequences_df.count()
    x_sequences_df.count()

    # ---------------- Prepare datasets ----------------
    print(f"{GRAY}Preparing training and evaluation datasets...{RESET}")

    x_train_normal_labelled = x_sequences_df.filter(col("Temp_label") == 0)
    X_train_all_data = x_sequences_df.filter(col("Temp_label") != 888)
    x_unlabeled_from_train = x_sequences_df.filter(col("Temp_label") == 999)

    labelled_df_from_train = sequences_df.filter(col("Temp_label") == 0)
    ground_truth_labeled_data_from_train = labelled_df_from_train.select("Label")

    labelled_df_from_train_all_data = sequences_df.filter(col("Temp_label") != 888)
    ground_truth_train_all_data = labelled_df_from_train_all_data.select("Label")

    unlabeled_df_from_train = sequences_df.filter(col("Temp_label") == 999)
    ground_truth_unlabeled_data_from_train = unlabeled_df_from_train.select("Label")

    unlabeled_df_from_test = sequences_df.filter(col("Temp_label") == 888)
    ground_truth_unlabeled_data_from_test = unlabeled_df_from_test.select("Label")

    labeled_df_from_val = sequences_df.filter(col("Temp_label") == 777)
    ground_truth_labeled_data_from_val = labeled_df_from_val.select("Label")

    # ---------------- Novelty detection ----------------
    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")

    X_train, y_train, X_test, y_test_truth, X_val, y_val_truth = \
        features_engineering_obj.novelty_detection_label_establishment(
            sequences_df,
            x_train_normal_labelled,
            x_unlabeled_from_train,
            ground_truth_unlabeled_data_from_train
        )

    # ---------------- Anomaly Detection ----------------
    print(f"{GRAY}Running anomaly detection on test dataset...{RESET}")

    y_test_truth, y_test_pred, fit_time, predict_time = \
        anomaly_detection_obj.anomaly_detector(
            X_train, y_train, X_test,
            y_test_truth, X_val, y_val_truth, mode
        )

    # ---------------- Model Evaluation ----------------
    print(f"{GRAY}Evaluating model performance...{RESET}")

    model_evaluation_obj.evaluation(
        Round,
        X_train,
        y_train,
        number_component,
        y_test_truth,
        y_test_pred,
        DATASET,
        X_test
    )

    print(f"Model training completed in {fit_time:.2f} minutes")
    print(f"Prediction completed in {predict_time:.2f} minutes")

    # ---------------- Cleanup ----------------
    sequences_df.unpersist()
    x_sequences_df.unpersist()
    final_train_with_test_with_val.unpersist()
    train_df.unpersist()
    val_df.unpersist()
    test_df.unpersist()

    spark.stop()





'''
# ====================== Main ======================
def main():
    # ---------------- Project configuration ----------------
    DATASET = 'HDFS'
    DATASETS_FOLDER = 'datasets'

    # Paths
    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}.csv'
    DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}_doc_topic'
    SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}_sentiment'
    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET}_All_pre_final_global_features.pkl'

    # ---------------- Initialize classes ----------------
    # logdata_read_obj = LogdataRead()
    features_extracting_obj = FeaturesExtractor()
    features_engineering_obj = FeaturesEngineering()
    anomaly_detection_obj = AnomalyDetector()
    model_evaluation_obj = ModelEvaluation()
    utilities_obj = Utilities()

    # ---------------- Dataset Splitting ----------------
    print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(ALL_DATASET_CSV_PATH, DATASET, spark)
    # ---------------- Process normal data ----------------
    print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
    final_train_with_test = utilities_obj.processing_data_portion(train_df, validate_df, test_df, df_features, spark)

    # ---------------- Features Extracting ----------------
    print(f"{GRAY}Extracting features for training and test datasets...{RESET}")
    number_component, best_topic_number, final_all_features_df = features_extracting_obj.features_extracting_configuring_tuning(
        features_extracting_obj, DOC_TOPIC_DF_PATH, SENTIMENT_DF_PATH, DATASET, PRE_FINAL_GLOBAL_FEATURES_PKL_PATH,
        final_train_with_test, spark)
    #    exit()

    print(' ----- start save parquet----------')
    final_all_features_df.cache()

    # ---------------- Features Engineering: Aggregation/Transformation ----------------
    print(f"{GRAY}Aggregating and transforming features...{RESET}")
    #    number_component=50
    output_path = DATASET + "_Topic_sentiment_diff_semantic_df.parquet"
    # ✅ Load from Parquet
    final_all_features_df = spark.read.parquet(output_path)
    final_all_features_df.printSchema()
    #    exit()

    print("✅ Loaded DataFrame schema:")

    sequences_df, x_sequences_df, y_sequences_df = features_engineering_obj.features_aggregation_transformation(
        number_component, final_all_features_df, DATASET, spark)
    #    exit()
    # ------------------- Feature splits -------------------

    # Features corresponding to Temp_label = 0 (normal labelled)
    x_train_normal_labelled = sequences_df.filter(col("Temp_label") == 0).select("features_vec_final")
    df_train_normal_labelled = sequences_df.filter(col("Temp_label") == 0).select("Node_block_id", "features_vec_final",
                                                                                  "Temp_label", "Label")

    # Features corresponding to Temp_label != 888 (all train data, labelled + unlabeled)
    X_train_all_data = sequences_df.filter(col("Temp_label") != 888).select("features_vec_final")

    # Features corresponding to Temp_label = 999 (unlabeled from train)
    x_unlabeled_from_train = sequences_df.filter(col("Temp_label") == 999).select("features_vec_final")

    # ------------------- Ground truth splits -------------------
    # Labelled train normal logs
    labelled_df_from_train = sequences_df.filter(col("Temp_label") == 0)
    ground_truth_labeled_data_from_train = labelled_df_from_train.select("Label")

    # Labelled + unlabeled train logs (Temp_label != 888)
    labelled_df_from_train_all_data = sequences_df.filter(col("Temp_label") != 888)
    ground_truth_train_all_data = labelled_df_from_train_all_data.select("Label")

    # Unlabeled train logs (Temp_label == 999)
    unlabeled_df_from_train = sequences_df.filter(col("Temp_label") == 999)
    df_unlabeled_from_train = sequences_df.filter(col("Temp_label") == 999).select("Node_block_id",
                                                                                   "features_vec_final", "Temp_label",
                                                                                   "Label")

    ground_truth_unlabeled_data_from_train = unlabeled_df_from_train.select("Label")

    # Unlabeled test logs (Temp_label == 888)
    unlabeled_df_from_test = sequences_df.filter(col("Temp_label") == 888)
    ground_truth_unlabeled_data_from_test = unlabeled_df_from_test.select("Label")

    # ---------------- Novelty detection and label establishment ----------------
    print(' start saving the final data .....')
    output_path = DATASET + "_sequences_df.parquet"
    sequences_df.write.mode("overwrite").parquet(output_path)
    print(f"✅ sequences_df saved successfully to {output_path}")
    exit()

    output_path = DATASET + "_sequences_df.parquet"
    # ✅ Load from Parquet
    sequences_df = spark.read.parquet(output_path)

    print("✅ Loaded DataFrame schema:")
    sequences_df.printSchema()
    number_component = 13

    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")
    df_final, df_test = features_engineering_obj.novelty_detection_label_establishment(sequences_df, method="gmm")
    #    exit()
    # ---------------- Anomaly Detection ----------------
    print(f"{GRAY}Running anomaly detection on test dataset...{RESET}")
    final_pred_df, df_final = anomaly_detection_obj.anomaly_detector(df_final, df_test, spark

                                                                     )

    # ---------------- Model Evaluation ----------------
    print(f"{GRAY}Evaluating model performance...{RESET}")
    model_evaluation_obj.evaluation(number_component, final_pred_df, df_final, DATASET)

    exit()

    # Load engineered features back into Spark
    final_train_with_test_pdf = pd.read_pickle(PRE_FINAL_GLOBAL_FEATURES_PKL_PATH)
    final_train_with_test = spark.createDataFrame(final_train_with_test_pdf)

    # ---------------- Features Engineering: Aggregation/Transformation ----------------
    print(f"{GRAY}Aggregating and transforming features...{RESET}")
    sequences_df, x_sequences_df, y_sequences_df = features_engineering_obj.features_aggregation_transformation(
        final_train_with_test, DATASET)

    # ---------------- Prepare datasets ----------------
    print(f"{GRAY}Preparing training and evaluation datasets...{RESET}")

    # NOTE: At this stage, if your feature engineering outputs Spark DataFrames,
    # you can use Spark filters instead of Pandas boolean indexing.
    # If they are still pandas/numpy (e.g., for ML training), leave as is.

    x_train_normal_labelled = x_sequences_df[y_sequences_df == 0]
    X_train_all_data = x_sequences_df[y_sequences_df != 888]
    x_unlabeled_from_train = x_sequences_df[y_sequences_df == 999]

    labelled_df_from_train = sequences_df.filter(sequences_df.Temp_label == 0)
    ground_truth_labeled_data_from_train = labelled_df_from_train.select("Label")

    labelled_df_from_train_all_data = sequences_df.filter(sequences_df.Temp_label != 888)
    ground_truth_train_all_data = labelled_df_from_train_all_data.select("Label")

    unlabeled_df_from_train = sequences_df.filter(sequences_df.Temp_label == 999)
    ground_truth_unlabeled_data_from_train = unlabeled_df_from_train.select("Label")

    unlabeled_df_from_test = sequences_df.filter(sequences_df.Temp_label == 888)
    ground_truth_unlabeled_data_from_test = unlabeled_df_from_test.select("Label")

    # ---------------- Novelty detection and label establishment ----------------
    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")
    x_train, y_train, x_test, y_test_truth = features_engineering_obj.novelty_detection_label_establishment(
        sequences_df, x_train_normal_labelled, x_unlabeled_from_train, ground_truth_unlabeled_data_from_train)

    # ---------------- Anomaly Detection ----------------
    print(f"{GRAY}Running anomaly detection on test dataset...{RESET}")
    y_test_truth, y_test_pred = anomaly_detection_obj.anomaly_detector(x_train, y_train, x_test, y_test_truth)

    # ---------------- Model Evaluation ----------------
    print(f"{GRAY}Evaluating model performance...{RESET}")
    model_evaluation_obj.evaluation(number_component, y_test_truth, y_test_pred, DATASET, x_test)


    # ---------------- Stop Spark ----------------
    spark.catalog.clearCache()
    spark.stop()

'''
if __name__ == "__main__":
    main()

