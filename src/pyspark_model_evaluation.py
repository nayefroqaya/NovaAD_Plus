import pandas as pd
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.linalg import Vectors
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from pyspark.sql.functions import col


class ModelEvaluation:
    """Class for evaluating model performance and feature importance."""
    @staticmethod
    def evaluation_pyspark(predictions_df):
        """
        Evaluate anomaly detection results using Spark DataFrame
        """

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        tp = predictions_df.filter((col("label") == 1) & (col("last_pred_label") == 1)).count()

        tn = predictions_df.filter((col("label") == 0) & (col("last_pred_label") == 0)).count()

        fp = predictions_df.filter((col("label") == 0) & (col("last_pred_label") == 1)).count()

        fn = predictions_df.filter((col("label") == 1) & (col("last_pred_label") == 0)).count()

        # -----------------------------
        # Metrics
        # -----------------------------
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

        # -----------------------------
        # Print results
        # -----------------------------
        print("\n📊 Evaluation Results")
        print("--------------------")
        print(f"TP: {tp}  FP: {fp}")
        print(f"FN: {fn}  TN: {tn}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"Accuracy  : {accuracy:.4f}")

        # -----------------------------
        # Return metrics (optional)
        # -----------------------------
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    @staticmethod
    def evaluation(number_components, final_pred_df, df_final, dataset):
        """
        Evaluate model performance and compute feature importance metrics.

        Parameters:
        number_components (int): Number of principal components used
        y_test_truth (array): Ground truth labels
        y_test_pred (array): Predicted labels
        dataset (str): Dataset identifier for output file naming
        X_test (DataFrame or array): Test features
        """

        # Print classification report
        # Convert required columns to pandas
        pdf = final_pred_df.select("label", "last_pred_label").toPandas()

        y_true = pdf["label"].values
        y_pred = pdf["last_pred_label"].values
        print('supervised final model performance ..........')
        report = classification_report(y_true, y_pred, digits=3)
        print(report)
        exit()

        # Define feature names
        feature_names = ([f"component_{i + 1}" for i in range(number_components)] + [  # Text features
            'sentiment', 'dominant_topic', 'word_count', 'character_count', 'entropy',

            # Temporal features
            'month', 'day', 'hour', 'minute', 'second'])

        num_features = df_final.select("features_vec_final").first()[0].size

        selector = UnivariateFeatureSelector(outputCol="selectedFeatures", labelCol="Final_Label",
                                             featuresCol="features_vec_final", selectionMode="numTopFeatures",
                                             selectionThreshold=num_features, # select ALL features
                                             featureType="continuous", labelType="categorical",
                                             selectorType="mutualInformation")

        model = selector.fit(df_final)
        mi_scores = model.getScores()  # returns an array of MI values
        # ============================================
        # 5. Build table of (index, name, score)
        # ============================================
        rows = [Row(feature_index=i, feature_name=feature_names[i], mi_score=float(mi_scores[i])) for i in
                range(len(mi_scores))]

        mi_df = spark.createDataFrame(rows)

        # ============================================
        # 6. Sort by MI importance
        # ============================================
        mi_df_sorted = mi_df.orderBy(mi_df.mi_score.desc())

        # ============================================
        # 7. Save to CSV
        # ============================================
        mi_df_sorted.coalesce(1).write.csv(dataset + "_mi_feature_importance", header=True, mode="overwrite")

        print("Mutual Information feature ranking saved to: mi_feature_importance/")
