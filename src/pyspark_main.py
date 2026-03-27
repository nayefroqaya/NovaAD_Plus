import colorama
import numpy as np
import os
import warnings
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import DataFrame

from pyspark_anomaly_detection import AnomalyDetector
from pyspark_features_engineering import FeaturesEngineering
from pyspark_features_extracting import FeaturesExtractor
from pyspark_model_evaluation import ModelEvaluation
from pyspark_utility import Utilities
from load_datalog import LogdataRead

from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import col
import psutil
import pandas as pd

import psutil
import platform
import pyspark
from pyspark.sql import SparkSession
import time

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


num_workers = 8
cores_per_worker = 4     # cores_per_worker = spark_cores // num_workers
executor_memory_gb = 20  # executor_memory_gb = spark_ram_gb // num_workers
driver_memory_gb = 20     # driver_memory_gb = executor_memory_gb
worker_memory_mib = executor_memory_gb * 1024   # worker_memory_mib = executor_memory_gb * 1024


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

print("Spark initialized")

'''

# ============================================================
# 1) System Info
# ============================================================
vm = psutil.virtual_memory()
total_gb = vm.total / (1024 ** 3)
available_gb = vm.available / (1024 ** 3)
used_gb = vm.used / (1024 ** 3)
percent_used = vm.percent

logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)

disk = psutil.disk_usage("/")
disk_total_gb = disk.total / (1024 ** 3)
disk_free_gb = disk.free / (1024 ** 3)

print("=== System Info ===")
print(f"Total RAM:     {total_gb:.2f} GB")
print(f"Available RAM: {available_gb:.2f} GB")
print(f"Used RAM:      {used_gb:.2f} GB ({percent_used:.1f}%)")
print(f"CPU cores (physical/logical): {physical_cores} / {logical_cores}")
print(f"Disk Total: {disk_total_gb:.2f} GB, Disk Free: {disk_free_gb:.2f} GB")
print(f"OS: {platform.platform()}")
print(f"Python version: {platform.python_version()}")
print(f"PySpark version: {pyspark.__version__}")

# ============================================================
# 2) Spark Cluster Emulation Settings
#    local-cluster[N,C,M]
#    N = number of workers
#    C = cores per worker
#    M = memory per worker in MiB
# ============================================================
num_workers = 4
cores_per_worker = 2
worker_memory_mib = 4096   # 4 GB per worker

# Total emulated executor cores
total_executor_cores = num_workers * cores_per_worker

# Keep driver/executor memory BELOW worker memory
driver_memory_gb = 4
executor_memory_gb = 3

# Shuffle partitions / parallelism
shuffle_partitions = total_executor_cores * 2
default_parallelism = total_executor_cores * 2

# Absolute directories are safer than relative ones
spill_dir = "/tmp/spark-spill"
eventlog_dir = "/tmp/spark-events"

os.makedirs(spill_dir, exist_ok=True)
os.makedirs(eventlog_dir, exist_ok=True)

print("\n=== Spark Config ===")
print(f"Master: local-cluster[{num_workers},{cores_per_worker},{worker_memory_mib}]")
print(f"Driver memory:   {driver_memory_gb}g")
print(f"Executor memory: {executor_memory_gb}g")
print(f"Total executor cores: {total_executor_cores}")
print(f"default.parallelism: {default_parallelism}")
print(f"shuffle.partitions:  {shuffle_partitions}")
print(f"Spill dir: {spill_dir}")
print(f"Event log dir: {eventlog_dir}")

# ============================================================
# 3) Initialize Spark
# ============================================================
spark = (
    SparkSession.builder
    .appName("AD")
    .master(f"local-cluster[{num_workers},{cores_per_worker},{worker_memory_mib}]")

    # -----------------------------
    # Memory
    # -----------------------------
    .config("spark.driver.memory", f"{driver_memory_gb}g")
    .config("spark.executor.memory", f"{executor_memory_gb}g")
    .config("spark.memory.fraction", "0.6")
    .config("spark.memory.storageFraction", "0.3")

    # -----------------------------
    # CPU / Parallelism
    # -----------------------------
    .config("spark.default.parallelism", str(default_parallelism))
    .config("spark.sql.shuffle.partitions", str(shuffle_partitions))

    # -----------------------------
    # Shuffle / Spill
    # -----------------------------
    .config("spark.reducer.maxSizeInFlight", "48m")
    .config("spark.shuffle.file.buffer", "32k")
    .config("spark.shuffle.spill.compress", "true")
    .config("spark.shuffle.compress", "true")
    .config("spark.local.dir", spill_dir)

    # -----------------------------
    # Adaptive Query Execution
    # -----------------------------
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")

    # -----------------------------
    # Logging / Monitoring
    # -----------------------------
    .config("spark.eventLog.enabled", "true")
    .config("spark.eventLog.dir", eventlog_dir)

    # -----------------------------
    # Serialization
    # -----------------------------
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    .getOrCreate()
)
'''

'''
# --------------------Start-All resources - default case 
# System Info
# -----------------------------
vm = psutil.virtual_memory()
total_gb = vm.total / (1024 ** 3)
available_gb = vm.available / (1024 ** 3)
used_gb = vm.used / (1024 ** 3)
percent_used = vm.percent

logical_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)

disk = psutil.disk_usage('/')
disk_total_gb = disk.total / (1024 ** 3)
disk_free_gb = disk.free / (1024 ** 3)

print("=== System Info ===")
print(f"Total RAM:     {total_gb:.2f} GB")
print(f"Available RAM: {available_gb:.2f} GB")
print(f"Used RAM:      {used_gb:.2f} GB ({percent_used:.1f}%)")
print(f"CPU cores (physical/logical): {physical_cores} / {logical_cores}")
print(f"Disk Total: {disk_total_gb:.2f} GB, Disk Free: {disk_free_gb:.2f} GB")
print(f"OS: {platform.platform()}")
print(f"Python version: {platform.python_version()}")
print(f"PySpark version: {pyspark.__version__}")

# -----------------------------
# Determine Spark Memory
# -----------------------------
spark_memory_gb = max(4, int(available_gb * 0.8))  # safer: 60% of available RAM
print(f"Setting Spark driver & executor memory to {spark_memory_gb} GB")

# -----------------------------
# Initialize Spark
# -----------------------------

num_cores = logical_cores   # or 4, 8, 16 for experiments

spark = (
    SparkSession.builder
    .appName("AD")
#    .master(f"local[{num_cores}]")
#    .master("local[*]")  # use all CPU cores
     .master("local-cluster[4,2,4096]")
    # -----------------------------
    # Memory Control (Core Experiment Variable)
    # -----------------------------
    .config("spark.driver.memory", f"{spark_memory_gb}g")
    .config("spark.executor.memory", f"{spark_memory_gb}g")
    .config("spark.executor.memoryOverhead", "4g")   # controls off-heap + shuffle buffers
    .config("spark.memory.fraction", "0.6")          # execution+storage memory
    .config("spark.memory.storageFraction", "0.3")   # cached data portion

    # -----------------------------
    # CPU / Parallelism Control
    # -----------------------------
    .config("spark.default.parallelism", str(num_cores * 2))
    .config("spark.sql.shuffle.partitions", str(num_cores * 2))

    # -----------------------------
    # Shuffle & Spill Control (KEY for your table)
    # -----------------------------
    .config("spark.reducer.maxSizeInFlight", "48m")  # smaller → more spill
    .config("spark.shuffle.file.buffer", "32k")      # smaller → more disk I/O
    .config("spark.shuffle.spill.compress", "true")
    .config("spark.shuffle.compress", "true")

    # -----------------------------
    # Disk Spill Location (for monitoring)
    # -----------------------------
    .config("spark.local.dir", "tmp/spark-spill")   # where spill files go

    # -----------------------------
    # Adaptive Query + Skew (for fairness)
    # -----------------------------
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")

    # -----------------------------
    # Spill & Shuffle Observability (for paper)
    # -----------------------------
    .config("spark.eventLog.enabled", "true")
    .config("spark.eventLog.dir", "tmp/spark-events")

    # -----------------------------
    # Serialization & Execution
    # -----------------------------
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()
)
'''

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
    DATASET = 'HDFS'
    DATASETS_FOLDER = 'datasets'
    round_id = '1'
    mode = 'X'
    Mix_or_stable = '0'

    # Paths
    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv'
    DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_doc_topic_df.pkl'
    SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_sentiment_df.pkl'
    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_pre_final_global_features.pkl'

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

    # ---------------- Data as CSV ----------------
    # logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)
    # print(' Reading the file was done successfully ')
    # exit()

    '''
    # ---------------- Load CSV into Spark ----------------
    all_data_df = spark.read.csv(ALL_DATASET_CSV_PATH, header=True, inferSchema=True).cache()
    all_data_df.count()  # Materialize cache
    print('✅ Loaded CSV into Spark DataFrame')

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
    #exit()
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
    print("Train count:", train_df.count())
    print("Validation count:", val_df.count())
    print("Test count:", test_df.count())
    #--------------Read Parquest file and convert to Pandas and save PKL
    train_pd = train_df.toPandas()
    val_pd = val_df.toPandas()
    test_pd = test_df.toPandas()
    train_pd.to_pickle(os.path.join(save_path, "train_df.pkl"))
    val_pd.to_pickle(os.path.join(save_path, "val_df.pkl"))
    test_pd.to_pickle(os.path.join(save_path, "test_df.pkl"))
    #----------------
    spark.stop()



    train_df.printSchema()
    assert train_df.schema == val_df.schema == test_df.schema

    exit()
    

    final_train_with_test_with_val = utilities_obj.processing_data_portion(train_df, val_df, test_df).persist(
        StorageLevel.MEMORY_AND_DISK)
    final_train_with_test_with_val.count()
    #exit()
    


    # ---------------- Features Extracting ----------------
    # --- Before Train ---
    shutil.rmtree(SPILL_DIR, ignore_errors=True)
    os.makedirs(SPILL_DIR, exist_ok=True)

    print(f"{GRAY}Extracting features for training and test datasets...{RESET}")
    start_features_extracting = time.time()
    number_component, best_topic_number, final_all_features_df = features_extracting_obj.features_extracting_configuring_tuning(round_id,
            features_extracting_obj,
            DOC_TOPIC_DF_PATH,
            SENTIMENT_DF_PATH,
            DATASET,
            PRE_FINAL_GLOBAL_FEATURES_PKL_PATH,
            final_train_with_test_with_val,spark
        )
    end_features_extracting= time.time()
    feature_extract_time = (end_features_extracting - start_features_extracting) / 60
    print(f"Model Features extracting completed in {feature_extract_time:.2f} minutes")

    extract_features_spill_gb = get_spill_size_gb(SPILL_DIR)
    print(f"Shuffle Spill during features extracting : {extract_features_spill_gb:.2f} GB")
    exit()


    # ---------------- Load feature PKL → Spark ----------------
    # ✅ Load from Parquet
    output_path = round_id + '_' + DATASET + "_Topic_sentiment_diff_semantic_df.parquet"
    final_train_with_test_with_val = spark.read.parquet(output_path)
    final_train_with_test_with_val.count()





    # ---------------- Features Engineering ----------------
    print(f"{GRAY}Aggregating and transforming features...{RESET}")
    shutil.rmtree(SPILL_DIR, ignore_errors=True)
    os.makedirs(SPILL_DIR, exist_ok=True)

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

    # sequences_df = sequences_df.persist(StorageLevel.MEMORY_AND_DISK)
    # x_sequences_df = x_sequences_df.cache()

    # sequences_df.count()
    # x_sequences_df.count()
    # exit()

    # ---------------- Prepare datasets ----------------
    print(f"{GRAY}Preparing training and evaluation datasets...{RESET}")
    # ---------------- Novelty detection ----------------
    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")

    shutil.rmtree(SPILL_DIR, ignore_errors=True)
    os.makedirs(SPILL_DIR, exist_ok=True)

    start_Novelty = time.time()
    df_final_train_cls, df_test_cls, df_val_cls = (
        features_engineering_obj.novelty_detection_label_establishment(DATASET, sequences_df=sequences_df, spark=spark,
                                                                       method="gmm"))

    Novelty_features_spill_gb = get_spill_size_gb(SPILL_DIR)
    print(f"Shuffle Spill during Novelty: {Novelty_features_spill_gb:.2f} GB")

    end_Novelty = time.time()
    Novelty_time = (end_Novelty - start_Novelty) / 60
    print(f"Model Novelty and label estimating completed in {Novelty_time:.2f} minutes")
    print('Novel was done .....')
    # exit()

    # ---------------- Anomaly Detection ----------------
    print(f"{GRAY}Running anomaly detection on test dataset...{RESET}")
    shutil.rmtree(SPILL_DIR, ignore_errors=True)
    os.makedirs(SPILL_DIR, exist_ok=True)

    start_anomaly = time.time()

    test_pred, LABEL_COL, best_thr, train_runtime_min, test_runtime_min = anomaly_detection_obj.anomaly_detector(
        df_train_quality, df_test_cls, df_val_cls, mode)
    test_metrics = model_evaluation_obj.evaluation_pyspark(test_pred, label_col=LABEL_COL, raw_pred_col="rawPrediction",
                                                           thr=best_thr, pos_index=1)

    print("\n=== TEST METRICS (LinearSVC single classifier) ===")
    print(f"Precision (anomaly=1): {test_metrics['P']:.4f}")
    print(f"Recall    (anomaly=1): {test_metrics['R']:.4f}")
    print(f"F1-score  (anomaly=1): {test_metrics['F1']:.4f}")
    print(f"TP={test_metrics['TP']} FP={test_metrics['FP']} FN={test_metrics['FN']}")
    print(f"[TIME] test_runtime_sec={test_runtime_min:.3f}")
    print(f"[TIME] train_runtime_sec={train_runtime_min:.3f}")

    exit()

    Anomaly_spill_gb = get_spill_size_gb(SPILL_DIR)
    print(f"Shuffle Spill during AD: {Anomaly_spill_gb:.2f} GB")
    end_anomaly = time.time()
    anomaly_time = (end_anomaly - start_anomaly) / 60
    print(f"Model anomaly train completed in {anomaly_time:.2f} minutes")

    # results["predictions_df"]  # Spark DF for evaluation
    # results["best_threshold"]  # chosen on validation
    # print(results["fit_time"])  # minutes
    # print(results["predict_time"])  # minutes

    # ---------------- Model Evaluation ----------------

    metrics = model_evaluation_obj.evaluation_pyspark(results["predictions_df"])
    print("\n📊 Final Test Metrics")
    print("--------------------")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
    print(results["fit_time"])  # minutes
    print(results["predict_time"])  # minutes
    # print(f"Model Features extracting completed in {feature_extract_time:.2f} minutes")
    # print(f"aggregation and transform completed in {start_agree_trans_time:.2f} minutes")
    # print(f"Model Novelty and label estimating completed in {Novelty_time:.2f} minutes")
    # print(f"Model anomaly train completed in {anomaly_time:.2f} minutes")
    #  print(f"Shuffle Spill during features extracting : {extract_features_spill_gb:.2f} GB")
    # print(f"Shuffle Spill during aggregation: {aggregation_features_spill_gb:.2f} GB")
    # print(f"Shuffle Spill during Novelty: {Novelty_features_spill_gb:.2f} GB")
    # print(f"Shuffle Spill during AD Train : {Anomaly_spill_gb:.2f} GB")

    exit()

    print(f"{GRAY}Evaluating model performance...{RESET}")

    model_evaluation_obj.evaluation(round_id, X_train, y_train, number_component, y_test_truth, y_test_pred, DATASET,
        X_test)

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
