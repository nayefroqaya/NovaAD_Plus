import warnings

import colorama
import pandas as pd
import torch

from utility import Utilities
from anomaly_detection import AnomalyDetector
from features_engineering import FeaturesEngineering
from features_extracting import FeaturesExtractor
from logdata_read import LogdataRead
from model_evaluation import ModelEvaluation

# ====================== Setup ======================
warnings.filterwarnings('ignore')
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


# ====================== Main ======================
def main():
    # ---------------- Device check ------------------
    if torch.cuda.is_available():
        print(f"{GREEN}GPU detected. Using GPU for encoding.{RESET}")
    else:
        print(f"{YELLOW}No GPU detected. Using CPU for encoding. Exiting program.{RESET}")
        exit()

    # ---------------- Display options ----------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # ---------------- Project configuration ----------------
    DATASET = 'UU_HDFS'
    DATASETS_FOLDER = 'datasets'

    # Paths
    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}.csv'
    DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}_All_doc_topic_df.pkl'
    SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}_All_sentiment_df.pkl'
    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET}_All_pre_final_global_features.pkl'

    # ---------------- Initialize classes ----------------
    logdata_read_obj = LogdataRead()
    features_extracting_obj = FeaturesExtractor()
    features_engineering_obj = FeaturesEngineering()
    anomaly_detection_obj = AnomalyDetector()
    model_evaluation_obj = ModelEvaluation()
    utilities_obj = Utilities()

    # ---------------- Data as CSV ----------------
    logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)
    print(' Reading the file was done successfully ')
    # ---------------- Dataset Splitting ----------------
    print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(
        ALL_DATASET_CSV_PATH, DATASET
    )

    # ---------------- Process normal data ----------------
    print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
    final_train_with_test = utilities_obj.processing_data_portion(
        train_df, validate_df, test_df, df_features
    )

    # ---------------- Features Extracting ----------------
    print(f"{GRAY}Extracting features for training and test datasets...{RESET}")
    number_component, best_topic_number = features_extracting_obj.features_extracting_configuring_tuning(
        features_extracting_obj,
        DOC_TOPIC_DF_PATH,
        SENTIMENT_DF_PATH,
        DATASET,
        PRE_FINAL_GLOBAL_FEATURES_PKL_PATH,
        final_train_with_test
    )

    final_train_with_test = pd.read_pickle(PRE_FINAL_GLOBAL_FEATURES_PKL_PATH)

    # ---------------- Features Engineering: Aggregation/Transformation ----------------
    print(f"{GRAY}Aggregating and transforming features...{RESET}")
    sequences_df, x_sequences_df, y_sequences_df = features_engineering_obj.features_aggregation_transformation(
        final_train_with_test,
        DATASET
    )

    # ---------------- Prepare datasets ----------------
    print(f"{GRAY}Preparing training and evaluation datasets...{RESET}")
    x_train_normal_labelled = x_sequences_df[y_sequences_df == 0]
    X_train_all_data = x_sequences_df[y_sequences_df != 888]
    x_unlabeled_from_train = x_sequences_df[y_sequences_df == 999]

    labelled_df_from_train = sequences_df[sequences_df['Temp_label'] == 0]
    ground_truth_labeled_data_from_train = labelled_df_from_train['Label']

    labelled_df_from_train_all_data = sequences_df[sequences_df['Temp_label'] != 888]
    ground_truth_train_all_data = labelled_df_from_train_all_data['Label']

    unlabeled_df_from_train = sequences_df[sequences_df['Temp_label'] == 999]
    ground_truth_unlabeled_data_from_train = unlabeled_df_from_train['Label']

    unlabeled_df_from_test = sequences_df[sequences_df['Temp_label'] == 888]
    ground_truth_unlabeled_data_from_test = unlabeled_df_from_test['Label']

    # ---------------- Novelty detection and label establishment ----------------
    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")
    x_train, y_train, x_test, y_test_truth = features_engineering_obj.novelty_detection_label_establishment(
        sequences_df,
        x_train_normal_labelled,
        x_unlabeled_from_train,
        ground_truth_unlabeled_data_from_train
    )

    # ---------------- Anomaly Detection ----------------
    print(f"{GRAY}Running anomaly detection on test dataset...{RESET}")
    y_test_truth, y_test_pred = anomaly_detection_obj.anomaly_detector(
        x_train, y_train, x_test, y_test_truth
    )

    # ---------------- Model Evaluation ----------------
    print(f"{GRAY}Evaluating model performance...{RESET}")
    model_evaluation_obj.evaluation(
        number_component, y_test_truth, y_test_pred, DATASET, x_test
    )


if __name__ == "__main__":
    main()
