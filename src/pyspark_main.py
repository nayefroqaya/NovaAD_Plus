import os
import time
import warnings

import colorama
import numpy as np
import pandas as pd
import psutil
import psutil
import pyspark
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import col
from pyspark.storagelevel import StorageLevel

from load_datalog import LogdataRead
from pyspark_anomaly_detection import AnomalyDetector
from pyspark_features_engineering import FeaturesEngineering
from pyspark_features_extracting import FeaturesExtractor
from pyspark_model_evaluation import ModelEvaluation
from pyspark_utility import Utilities

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = str

    # ============================================================  # Detect system resources  # ============================================================
logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)

vm = psutil.virtual_memory()
total_ram_gb = vm.total // (1024 ** 3)

print("Detected CPU cores:", logical_cores)
print("Detected RAM:", total_ram_gb, "GB")

# ============================================================
# Spark cluster sizing
# ============================================================


num_workers = 8  # defualt =8
cores_per_worker =4   # defaul =4  # cores_per_worker = spark_cores // num_workers
executor_memory_gb =25 # executor_memory_gb = spark_ram_gb // num_workers
driver_memory_gb = 25  # driver_memory_gb = executor_memory_gb
worker_memory_mib = executor_memory_gb * 1024  # worker_memory_mib = executor_memory_gb * 1024

total_executor_cores = num_workers * cores_per_worker
shuffle_partitions = total_executor_cores * 2
default_parallelism = total_executor_cores * 2

# use half the CPU cores for Spark
spark_cores = logical_cores // 2

# allocate ~60% RAM to Spark
spark_ram_gb = int(total_ram_gb * 0.6)

# ============================================================
# directories
# ============================================================

spill_dir = "/tmp/spark-spill"
eventlog_dir = "/tmp/spark-events"

os.makedirs(spill_dir, exist_ok=True)
os.makedirs(eventlog_dir, exist_ok=True)

# ============================================================
# Start Spark
# ============================================================

spark = (SparkSession.builder.appName("Distributed_Log_AD")

         .master(f"local-cluster[{num_workers},{cores_per_worker},{worker_memory_mib}]")

         # memory
         .config("spark.driver.memory", f"{driver_memory_gb}g").config("spark.executor.memory",
                                                                       f"{executor_memory_gb}g")

         .config("spark.memory.fraction", "0.6").config("spark.memory.storageFraction", "0.3")

         # parallelism
         .config("spark.default.parallelism", default_parallelism).config("spark.sql.shuffle.partitions",
                                                                          shuffle_partitions)

         # shuffle
         .config("spark.reducer.maxSizeInFlight", "48m").config("spark.shuffle.file.buffer", "32k")

         # disk spill
         .config("spark.local.dir", spill_dir)

         # adaptive execution
         .config("spark.sql.adaptive.enabled", "true").config("spark.sql.adaptive.coalescePartitions.enabled", "true")

         # logging
         .config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", eventlog_dir)

         # serialization
         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")


         .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")
print("Spark initialized")



# ===================== ======================
warnings.filterwarnings('ignore')
colorama.init()
# exit()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW
import os
import shutil

SPILL_DIR = "/storage/home/roqaya/NovaAD_Plus/spark-spill"
os.makedirs(SPILL_DIR, exist_ok=True)


def get_spill_size_gb(spill_dir):
    total_bytes = 0
    for root, dirs, files in os.walk(spill_dir):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    return total_bytes / (1024 ** 3)


def main():
    # ---------------- Device setup ----------------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.enabled = True

    # ---------------- Display options ----------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # ---------------- Project configuration ----------------
    DATASET = 'TH_1G_ratio'
    DATASETS_FOLDER = 'datasets'
    round_id = '1'
    mode = 'X'
    Mix_or_stable = '0'

    # Paths
    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv'

    DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{round_id}_{DATASET}_All_doc_topic_df.pkl'
    SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{round_id}_{DATASET}_All_sentiment_df.pkl'
    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{round_id}_{DATASET}_All_pre_final_global_features.pkl'

    # ---------------- Spark session ----------------
    # spark = SparkSession.builder \
    #    .appName("LogAnomalyPipeline") \
    #    .getOrCreate()

    # ---------------- Initialize classes ----------------
    logdata_read_obj = LogdataRead()
    features_extracting_obj = FeaturesExtractor()
    features_engineering_obj = FeaturesEngineering()
    anomaly_detection_obj = AnomalyDetector()
    model_evaluation_obj = ModelEvaluation()
    utilities_obj = Utilities()

    '''
    # ---------------- Data as CSV ----------------
    logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)
    print(' Reading the file was done successfully ')
    #spark.stop()
    #exit()



    # ---------------- Load CSV into Spark ----------------
    all_data_df = spark.read.csv(ALL_DATASET_CSV_PATH, header=True, inferSchema=True).cache()
    all_data_df.count()  # Materialize cache
    print('✅ Loaded CSV into Spark DataFrame')
    #df1 = all_data_df.query("Label == '-'").reset_index(drop=True)  # Normal logs
    #df2 = all_data_df.query("Label != '-'").reset_index(drop=True)  # Anomaly logs

    # ---------------- Dataset Splitting ----------------
    print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(all_data_df, DATASET, round_id,
                                                                                  Mix_or_stable, spark)
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    validate_df = validate_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)
    train_df.count()
    validate_df.count()
    test_df.count()
    # exit()
    '''


    # ---------------- Process normal data ----------------
    if Mix_or_stable == '0' and DATASET == 'S_BGL':
        save_path = f"../datasets/{DATASET}/{round_id}_{DATASET}_Stable_Splitted_Datasets"

    elif Mix_or_stable == '1' and DATASET == 'S_BGL':
        save_path = f"../datasets/{DATASET}/{round_id}_{DATASET}_Mix_Splitted_Datasets"

    else:
        save_path = f"../datasets/{DATASET}/{round_id}_{DATASET}_Splitted_Datasets"

    os.makedirs(save_path, exist_ok=True)

    # ---------------- Load PKL splits into Spark ----------------
    train_df = spark.read.parquet(os.path.join(save_path, "train_df")).cache()
    val_df = spark.read.parquet(os.path.join(save_path, "val_df")).cache()
    test_df = spark.read.parquet(os.path.join(save_path, "test_df")).cache()
    # print("Train count:", train_df.count())
    # print("Validation count:", val_df.count())
    # print("Test count:", test_df.count())
    train_df.printSchema()
    val_df.printSchema()
    test_df.printSchema()
    spark.stop()
    exit()

    #train_df.select("Original_Label").distinct().show()
    #train_df.select("Label").distinct().show()

    #val_df.select("Original_Label").distinct().show()
    #val_df.select("Label").distinct().show()

    #test_df.select("Original_Label").distinct().show()
    #test_df.select("Label").distinct().show()

    # --------------Read Parquest file and convert to Pandas and save PKL for experiments with baseline
    train_pd = train_df.toPandas()
    val_pd = val_df.toPandas()
    test_pd = test_df.toPandas()
    train_pd.to_pickle(os.path.join(save_path,round_id + '_'+DATASET+ '_'+ "train_df.pkl"))
    val_pd.to_pickle(os.path.join(save_path,round_id + '_'+DATASET+ '_'+ "val_df.pkl"))
    test_pd.to_pickle(os.path.join(save_path, round_id + '_'+DATASET+ '_'+"test_df.pkl"))
    # ----------------
    train_df.printSchema()
    assert train_df.schema == val_df.schema == test_df.schema
    spark.stop()

    exit()

    final_train_with_test_with_val = utilities_obj.processing_data_portion(train_df, val_df, test_df).persist(
        StorageLevel.MEMORY_AND_DISK)
    final_train_with_test_with_val.count()


    # ---------------- Features Extracting ----------------
    # --- Before Train ---
    #shutil.rmtree(SPILL_DIR, ignore_errors=True)
    #os.makedirs(SPILL_DIR, exist_ok=True)

    print(f"{GRAY}Extracting features for training and test datasets...{RESET}")
    start_features_extracting = time.time()
    number_component, best_topic_number, final_all_features_df = features_extracting_obj.features_extracting_configuring_tuning(
        round_id,  DATASET,
        PRE_FINAL_GLOBAL_FEATURES_PKL_PATH, final_train_with_test_with_val, spark)
    end_features_extracting = time.time()
    feature_extract_time = (end_features_extracting - start_features_extracting) / 60
    print(f"Model Features extracting completed in {feature_extract_time:.2f} minutes")

    #extract_features_spill_gb = get_spill_size_gb(SPILL_DIR)
    #print(f"Shuffle Spill during features extracting : {extract_features_spill_gb:.2f} GB")
    #spark.stop()
    #exit()





    

    # ---------------- Load feature PKL → Spark ----------------
    # ✅ Load from Parquet
    output_path = f'../{DATASETS_FOLDER}/{DATASET}/{round_id}_{DATASET}_Topic_sentiment_diff_semantic_df.parquet'  #round_id + '_' + DATASET + "_Topic_sentiment_diff_semantic_df.parquet"
    final_train_with_test_with_val = spark.read.parquet(output_path)
    final_train_with_test_with_val.count()
    final_train_with_test_with_val.printSchema()
    final_train_with_test_with_val.select("Label").distinct().show()

    # spark.stop()
    #exit()

    # ---------------- Features Engineering ----------------
    #print(f"{GRAY}Aggregating and transforming features...{RESET}")
    #shutil.rmtree(SPILL_DIR, ignore_errors=True)
    #os.makedirs(SPILL_DIR, exist_ok=True)

    start_agree_trans = time.time()

    sequences_df, x_sequences_df, y_sequences_df = features_engineering_obj.features_aggregation_transformation(
        final_train_with_test_with_val, DATASET)
    print(
        sequences_df.columns)  # ['Node_block_id', 'features', 'Label', 'Temp_label', 'features_vec', 'features_vec_final']

    # exit()

    aggregation_features_spill_gb = get_spill_size_gb(SPILL_DIR)
    print(f"Shuffle Spill during aggregation: {aggregation_features_spill_gb:.2f} GB")
    # exit()

    end_agree_trans = time.time()
    start_agree_trans_time = (end_agree_trans - start_agree_trans) / 60
    print(f"aggregation and transform completed in {start_agree_trans_time:.2f} minutes")


    # ---------------- Prepare datasets ----------------
    print(f"{GRAY}Preparing training and evaluation datasets...{RESET}")
    # ---------------- Novelty detection ----------------
    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")

    shutil.rmtree(SPILL_DIR, ignore_errors=True)
    os.makedirs(SPILL_DIR, exist_ok=True)

    # start_Novelty = time.time()

    df_full_train_labeled_features, sequences_df = features_engineering_obj.novelty_detection_label_establishment(
        DATASET, sequences_df, spark)

    # ---------------- Anomaly Detection ----------------

    case1_test_pdf, case2_test_pdf, case1_classification_time, case1_Classification_pred_time, case2_Classification_time, case2_Classification_pred_time = anomaly_detection_obj.anomaly_detector(
        df_full_train_labeled_features, sequences_df)

    # ---------------- Evaluation----------------

    model_evaluation_obj.evaluation_pyspark(case1_test_pdf, case2_test_pdf, case1_classification_time,
                                            case1_Classification_pred_time, case2_Classification_time,
                                            case2_Classification_pred_time)

    spark.stop()



if __name__ == "__main__":
    main()
