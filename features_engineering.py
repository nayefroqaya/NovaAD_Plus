import warnings
from itertools import product

import colorama
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class FeaturesEngineering:

    @staticmethod
    def summing_Train_test_together(dataset, df_train_with_test):
        df_train_with_test = df_train_with_test.dropna(how='all')
        df_train_with_test = df_train_with_test.reset_index(drop=True)

        # ------- Train/Test dataset: --------------------------------------------------------------------------
        log_normal_labelled = df_train_with_test[(df_train_with_test['Temp_label'] == 0)]  # Normals logs - labeled
        log_remain_normal_anomaly_unlabelled = df_train_with_test[
            (df_train_with_test['Temp_label'] == 999)]  # Normal/Anomaly logs - unlabeled train data
        log_test_unlabelled = df_train_with_test[
            (df_train_with_test['Temp_label'] == 888)]  # Normal/Anomaly logs - unlabeled test data

        # (1) Train Normal logs labelled -----------------------------------------------------------------------
        summed_df_normal_labelled_train = log_normal_labelled.groupby('Node_block_id')['features'].apply(
            lambda x: np.mean(np.stack([np.asarray(i, dtype=np.float32) for i in x]), axis=0)
        ).reset_index()

        # Assigning a label to each log_sequence_id
        if dataset == 'UU_HDFS':
            sequence_labels_normal_labelled = log_normal_labelled.groupby('Node_block_id')['Label'].apply(
                lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()
        else:
            sequence_labels_normal_labelled = log_normal_labelled.groupby('Node_block_id')['Label'].apply(
                lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()

        # Merging the summed feature vectors with their respective labels
        summed_df_normal_labelled_train = summed_df_normal_labelled_train.merge(sequence_labels_normal_labelled,
                                                                                on='Node_block_id')
        summed_df_normal_labelled_train['Temp_label'] = 0  # Normal

        # Validation checks
        xx_anomaly = summed_df_normal_labelled_train[(summed_df_normal_labelled_train['Label'] == 'anomaly')]
        yy_normal = summed_df_normal_labelled_train[(summed_df_normal_labelled_train['Label'] == 'normal')]

        if len(xx_anomaly) != 0 or len(yy_normal) == 0:
            print('Error in summing_Train_test_together function: Normal data validation failed')
            exit()

        if len(yy_normal) != len(summed_df_normal_labelled_train):
            print('Error: Length mismatch in normal labelled data')
            exit()

        # (2) Train - Normal and Abnormal logs unlabelled ------------------------------------------------------
        summed_df_combine_unlabelled_train = log_remain_normal_anomaly_unlabelled.groupby('Node_block_id')[
            'features'].apply(
            lambda x: np.mean(np.stack(x), axis=0)).reset_index()

        if dataset == 'UU_HDFS':
            sequence_labels_combine_unlabelled = log_remain_normal_anomaly_unlabelled.groupby('Node_block_id')[
                'Label'].apply(lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()
        else:
            sequence_labels_combine_unlabelled = log_remain_normal_anomaly_unlabelled.groupby('Node_block_id')[
                'Label'].apply(lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()

        # Merging the summed feature vectors with their respective labels
        summed_df_combine_unlabelled_train = summed_df_combine_unlabelled_train.merge(
            sequence_labels_combine_unlabelled,
            on='Node_block_id')
        summed_df_combine_unlabelled_train['Temp_label'] = 999  # Normal and anomaly unlabeled

        # Validation checks
        xx_anomaly = summed_df_combine_unlabelled_train[(summed_df_combine_unlabelled_train['Label'] == 'anomaly')]
        yy_normal = summed_df_combine_unlabelled_train[(summed_df_combine_unlabelled_train['Label'] == 'normal')]

        if len(xx_anomaly) == 0 or len(yy_normal) == 0:
            print('Error in summing_Train_test_together function: Unlabeled data validation failed')
            exit()

        summed_df_train = pd.concat([summed_df_normal_labelled_train, summed_df_combine_unlabelled_train])

        # (3) Test dataset: ------------------------------------------------------------------------------------
        summed_df_combine_unlabelled_test = log_test_unlabelled.groupby('Node_block_id')['features'].apply(
            lambda x: np.mean(np.stack(x), axis=0)).reset_index()

        if dataset == 'UU_HDFS':
            sequence_labels_combine_unlabelled_test = log_test_unlabelled.groupby('Node_block_id')[
                'Label'].apply(lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()
        else:
            sequence_labels_combine_unlabelled_test = log_test_unlabelled.groupby('Node_block_id')[
                'Label'].apply(lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()

        # Merging the summed feature vectors with their respective labels
        summed_df_combine_unlabelled_test = summed_df_combine_unlabelled_test.merge(
            sequence_labels_combine_unlabelled_test,
            on='Node_block_id')
        summed_df_combine_unlabelled_test['Temp_label'] = 888  # Test normal and anomaly unlabeled

        summed_df_test = summed_df_combine_unlabelled_test

        # Final processing and combination
        summed_df_train = summed_df_train.reset_index(drop=True)
        summed_df_test = summed_df_test.reset_index(drop=True)

        summ_train_test_combine = pd.concat([summed_df_train, summed_df_test])
        summ_train_test_combine = summ_train_test_combine.reset_index(drop=True)

        print('Feature aggregation completed successfully')
        return summ_train_test_combine

    @staticmethod
    def preparing_features_training_combined_labeled_unlabeled(dataset, df_features_all_train_with_test_final):
        df_features_all_train_with_test_final['features'] = df_features_all_train_with_test_final['reduced_embedding']

        # Append additional features to the feature vector
        feature_columns = ['sentiment_label', 'Dominant_Topic', 'num_words', 'Character_Count',
                           'entropy', 'month', 'day', 'hour', 'minute', 'second']

        for col in feature_columns:
            df_features_all_train_with_test_final['features'] = df_features_all_train_with_test_final.apply(
                lambda row: row['features'] + [row[col]], axis=1)

        return df_features_all_train_with_test_final

    @staticmethod
    def features_aggregation_transformation(final_train_with_test, dataset):
        # Function to handle replacement of None, NaN, or other values
        def replace_none_or_nan(x):
            if x is None:
                return np.nan
            if isinstance(x, list):
                return [np.nan if elem in ['', 'None', 'NaN'] else elem for elem in x]
            if isinstance(x, (str, int, float)) and x in ['', 'None', 'NaN']:
                return np.nan
            return x

        final_train_with_test = final_train_with_test.applymap(replace_none_or_nan)
        final_train_with_test = final_train_with_test.dropna(how='all')
        final_train_with_test = final_train_with_test.reset_index(drop=True)

        # Optimize data types
        final_train_with_test = final_train_with_test.astype(
            {col: 'float32' if final_train_with_test[col].dtype == 'float64'
            else 'int32' if final_train_with_test[col].dtype == 'int64'
            else final_train_with_test[col].dtype for col in final_train_with_test.columns}
        )

        # Encode sentiment labels
        label_encoder_sentiment = LabelEncoder()
        final_train_with_test['sentiment_label'] = label_encoder_sentiment.fit_transform(
            final_train_with_test['sentiment_label'])

        # Convert embeddings to lists
        final_train_with_test['reduced_embedding'] = final_train_with_test['reduced_embedding'].apply(
            lambda x: x.tolist())

        # Prepare features and aggregate
        df_train_with_test = FeaturesEngineering.preparing_features_training_combined_labeled_unlabeled(
            dataset, final_train_with_test)
        sequences_df = FeaturesEngineering.summing_Train_test_together(dataset, df_train_with_test)

        # Apply Standard Scaler to the summed feature vectors
        Train_Test_df_lst = np.vstack(sequences_df['features'].values)
        sequences_df['Label'] = sequences_df['Label'].apply(lambda x: 0 if x == 'normal' else 1)

        # Validation checks
        xx_0 = sequences_df[(sequences_df['Label'] == 0)]
        xx_1 = sequences_df[(sequences_df['Label'] == 1)]
        if len(xx_0) == 0 or len(xx_1) == 0:
            print('Error: Truth label distribution issue')
            exit()

        scaler_data = StandardScaler()
        scaled_Train_Test_df = scaler_data.fit_transform(Train_Test_df_lst)
        print('Feature scaling completed')

        # Convert features into a NumPy array
        X_sequences_df = scaled_Train_Test_df
        y_sequences_df = sequences_df['Temp_label']

        return sequences_df, X_sequences_df, y_sequences_df

    @staticmethod
    def novelty_detection_label_establishment(sequences_df, X_train_normal_labelled, X_unlabeled_train,
                                              ground_truth_unlabeled_data_from_train):
        # Parameter grid for One-Class SVM
        gamma_values = np.linspace(0.2, 0.5, 5)
        nu_values = np.linspace(0.01, 0.08, 5)

        best_params = None
        best_score = -np.inf  # Higher LOF separation score is better

        # Grid Search Over Gamma & Nu
        for gamma, nu in product(gamma_values, nu_values):
            print(f'Testing parameters: Gamma={gamma}, Nu={nu}')

            # Train One-Class SVM ONLY on Normal Labeled Data
            oc_svm = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
            oc_svm.fit(X_train_normal_labelled)

            # Predict Labels on Unlabeled Data (1 = normal, -1 = anomaly)
            preds = oc_svm.predict(X_unlabeled_train)

            # Compute LOF Scores (Only for Points Classified as Normal)
            lof = LocalOutlierFactor(n_neighbors=20)
            lof_scores = -lof.fit_predict(X_unlabeled_train[preds == 1])  # Higher values = anomalies

            # Compute Silhouette Score
            if len(set(preds)) > 1:
                silhouette = silhouette_score(X_unlabeled_train, preds)
            else:
                silhouette = -1  # Invalid case (all one class)

            # Hybrid Metric = Weighted Sum of LOF & Silhouette
            hybrid_score = 0.5 * np.mean(lof_scores) + 0.5 * silhouette

            # Select Best Gamma & Nu
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_params = (gamma, nu)

        print(f"Optimal parameters found: Gamma={best_params[0]}, Nu={best_params[1]}")

        # Train final model with optimal parameters
        oc_svm = OneClassSVM(kernel='rbf', gamma=best_params[0], nu=best_params[1])
        oc_svm.fit(X_train_normal_labelled)

        # Make predictions
        y_pred = oc_svm.predict(X_unlabeled_train)
        pseudo_labels = np.where(y_pred == -1, 1, 0)  # Map -1 (outliers) to 1 (anomaly), 1 (inliers) to 0 (normal)

        # Ensure matching lengths
        if len(ground_truth_unlabeled_data_from_train) != len(pseudo_labels):
            print("Error: Length mismatch between ground truth and predictions")
            return None

        # Print classification report for pseudo-labels
        print("\nPseudo-Labeling Classification Report (One-Class SVM):")
        print(classification_report(ground_truth_unlabeled_data_from_train, pseudo_labels,
                                    target_names=["Normal", "Anomaly"]))

        # Prepare the Training Data ----------------------------------------------------------------------------
        df_train_normal = sequences_df[sequences_df['Temp_label'] == 0][['features', 'Temp_label', 'Label']].copy()
        df_train_normal['Final_Label'] = df_train_normal['Label']
        df_train_normal = df_train_normal[['features', 'Temp_label', 'Label', 'Final_Label']]

        df_train_unlabeled = sequences_df[sequences_df['Temp_label'] == 999][
            ['features', 'Temp_label', 'Label']].copy()
        df_train_unlabeled['Final_Label'] = pseudo_labels
        df_train_unlabeled = df_train_unlabeled[['features', 'Temp_label', 'Label', 'Final_Label']]

        # Combine both into a single DataFrame
        df_final = pd.concat([df_train_normal, df_train_unlabeled], ignore_index=True)

        # Final evaluation
        y_true = df_final['Label'].values
        y_pred = df_final['Final_Label'].values
        print("\nFull Training Data Classification Report (One-Class SVM):")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

        # Prepare test data
        df_test = sequences_df[sequences_df['Temp_label'] == 888][['features', 'Temp_label', 'Label']].copy()
        df_test = df_test[['features', 'Temp_label', 'Label']]

        # Validation check
        xx_0 = df_test[(df_test['Label'] == 0)]
        xx_1 = df_test[(df_test['Label'] == 1)]
        if len(xx_0) == 0 or len(xx_1) == 0:
            print('Error: Test data label distribution issue')
            exit()

        # Extract features and labels
        X_train = df_final['features'].tolist()
        y_train = df_final['Final_Label'].values
        X_test = df_test['features'].tolist()
        y_test_truth = df_test['Label'].values

        return X_train, y_train, X_test, y_test_truth
