import warnings
from datetime import datetime
import colorama
import numpy as np
import pandas as pd

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
    def insert_rows(df, row):
        """Insert a row at the end of the DataFrame."""
        insert_loc = df.index.max()
        if pd.isna(insert_loc):
            df.loc[0] = row
        else:
            df.loc[insert_loc + 1] = row

    @staticmethod
    def clean_up_df(df_features):
        """Remove rows with null content and display summary."""
        df_features = df_features[df_features['Content'].notnull()]
        print(GREEN + "[INFO] DataFrame cleaned. Null 'Content' rows removed." + RESET)
        df_features.info()
        return df_features

    @staticmethod
    def dataset_splitting(All_dataset_path_as_csv, dataset):
        """Load dataset CSV and split into train, validation, and test sets."""
        print(GREEN + f"[INFO] Preparing dataset '{dataset}'..." + RESET)
        df_features = pd.read_csv(All_dataset_path_as_csv, escapechar='\\')
        df_features.info()

        # Clean data
        df_features = Utilities.clean_up_df(df_features)

        # Standardize timestamps
        def update_timestamp(original_timestamp, desired_format='%Y-%m-%d %H:%M:%S.%f'):
            """Ensure timestamp matches the desired format."""
            try:
                datetime.strptime(str(original_timestamp), desired_format)
                return original_timestamp
            except ValueError:
                original_format = '%Y-%m-%d %H:%M:%S'
                parsed_timestamp = datetime.strptime(str(original_timestamp), original_format)
                if '%f' in desired_format and parsed_timestamp.microsecond == 0:
                    parsed_timestamp = parsed_timestamp.replace(microsecond=0)
                return parsed_timestamp.strftime(desired_format)

        df_features['Timestamp'] = df_features['Timestamp'].apply(update_timestamp)
        df_features.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)
        df_features.reset_index(drop=True, inplace=True)
        df_features = df_features[['Timestamp', 'Date', 'Time', 'Content',
                                   'processed_EventTemplate', 'Node_block_id', 'Label']]
        print(GREEN + "[INFO] Dataset timestamps standardized and sorted." + RESET)

        # Split dataset based on dataset type
        unique_ids = df_features['Node_block_id'].unique()
        total_ids = len(unique_ids)

        if dataset == 'UU_HDFS':
            # HDFS split: 60% train, 10% validation, 30% test
            train_end = int(0.6 * total_ids)
            val_end = train_end + int(0.1 * total_ids)
            train_ids, val_ids, test_ids = unique_ids[:train_end], unique_ids[train_end:val_end], unique_ids[val_end:]

        elif dataset in ['UU_BGL', 'UU_TH']:
            # Shuffle and split
            shuffled_ids = np.random.permutation(unique_ids)
            train_size, val_size = int(0.6 * total_ids), int(0.1 * total_ids)
            train_ids, val_ids, test_ids = shuffled_ids[:train_size], shuffled_ids[train_size:train_size + val_size], shuffled_ids[train_size + val_size:]
        else:
            raise ValueError(f"[ERROR] Unsupported dataset type: {dataset}")

        # Check for overlaps between splits
        set_train, set_val, set_test = set(train_ids), set(val_ids), set(test_ids)
        intersections = {
            "train_val": set_train.intersection(set_val),
            "train_test": set_train.intersection(set_test),
            "val_test": set_val.intersection(set_test)
        }
        for k, v in intersections.items():
            print(YELLOW + f"[CHECK] Intersection {k}: {v}" + RESET)

        if all(len(v) == 0 for v in intersections.values()):
            print(GREEN + "[INFO] No overlaps found between train, validation, and test sets." + RESET)
        else:
            raise ValueError("[ERROR] Overlaps detected between dataset splits!")

        # Create DataFrames for splits
        train_df = df_features[df_features['Node_block_id'].isin(train_ids)].copy()
        train_df['Type_ds'] = 'Train'
        val_df = df_features[df_features['Node_block_id'].isin(val_ids)].copy()
        val_df['Type_ds'] = 'Validation'
        test_df = df_features[df_features['Node_block_id'].isin(test_ids)].copy()
        test_df['Type_ds'] = 'Test'

        # Display split info
        print(GREEN + f"[INFO] Dataset split complete. Sizes -> Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}" + RESET)
        return train_df, val_df, test_df, df_features

    @staticmethod
    def processing_data_portion(train_df, validate_df, test_df, df_features):
        """Create labeled and unlabeled portions for training and mark test set in PySpark."""

        unique_normal = [r.Node_block_id for r in
                         train_df.filter(F.col("Label") == "Normal").select("Node_block_id").distinct().collect()]
        unique_anomaly = [r.Node_block_id for r in
                          train_df.filter(F.col("Label") == "Anomaly").select("Node_block_id").distinct().collect()]

        print(GREEN + f"[INFO] Total unique normal Node_block_ids: {len(unique_normal)}" + RESET)
        print(GREEN + f"[INFO] Total unique anomaly Node_block_ids: {len(unique_anomaly)}" + RESET)

        # Select 50% of normal blocks for labeled training
        selected_normal_50 = np.random.choice(unique_normal, size=len(unique_normal) // 2, replace=False).tolist()

        df_train_normal_50 = train_df.filter(F.col("Node_block_id").isin(selected_normal_50)).withColumn("Temp_label",
                                                                                                         F.lit(0))

        # Remaining normal + all anomaly blocks are unlabeled
        remaining_normal = list(set(unique_normal) - set(selected_normal_50))
        df_train_unlabeled = train_df.filter(
            (F.col("Node_block_id").isin(remaining_normal)) | (F.col("Node_block_id").isin(unique_anomaly))
        ).withColumn("Temp_label", F.lit(999))

        # Mark test set with Temp_label = 888
        test_df = test_df.withColumn("Temp_label", F.lit(888))

        # Combine all datasets
        final_dataset = df_train_normal_50.union(df_train_unlabeled).union(test_df)

        print(
            GREEN + f"[INFO] Labeled normal blocks: {df_train_normal_50.select('Node_block_id').distinct().count()} (Temp_label=0)" + RESET)
        print(
            GREEN + f"[INFO] Unlabeled blocks (remaining normal + anomaly): {df_train_unlabeled.select('Node_block_id').distinct().count()} (Temp_label=999)" + RESET)
        print(
            GREEN + f"[INFO] Test blocks: {test_df.select('Node_block_id').distinct().count()} (Temp_label=888)" + RESET)
        print(GREEN + f"[INFO] Final combined dataset size: {final_dataset.count()} rows" + RESET)

        final_dataset.printSchema()
        return final_dataset
