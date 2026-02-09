import pandas as pd
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.linalg import Vectors
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from pyspark.sql.functions import col

import warnings
import colorama
from pyspark.sql import functions as F

from pyspark.sql.functions import col, when, lit, udf
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col, when, lit
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import time
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.functions import vector_to_array
# ✅ Alias Spark ML classes to avoid ANY shadowing / UnboundLocalError
from pyspark.ml.classification import (
    LogisticRegression as SparkLogisticRegression,
    RandomForestClassifier as SparkRandomForestClassifier,
    GBTClassifier as SparkGBTClassifier,
)

class ModelEvaluation:
    """Class for evaluating model performance and feature importance."""

    def prf_at_threshold_fast(scored_df, thr, label_col="y", score_col="s1"):
        tmp = scored_df.select(col(label_col).alias("y"),
                               when(col(score_col) >= lit(thr), 1).otherwise(0).alias("yhat"))
        agg = tmp.agg(F.sum(((col("yhat") == 1) & (col("y") == 1)).cast("int")).alias("tp"),
                      F.sum(((col("yhat") == 1) & (col("y") == 0)).cast("int")).alias("fp"),
                      F.sum(((col("yhat") == 0) & (col("y") == 1)).cast("int")).alias("fn"), ).collect()[0]

        tp, fp, fn = int(agg["tp"]), int(agg["fp"]), int(agg["fn"])
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        return float(p), float(r), float(f1), tp, fp, fn


    @staticmethod

    def  evaluation_pyspark(predictions_df, label_col, raw_pred_col="rawPrediction", thr=0.0,
                                   pos_index=1):
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql import functions as F
        """
        predictions_df: output of model.transform(df)
        Uses rawPrediction[pos_index] as score (s1), thresholds at thr, returns metrics dict.
        """
        scored = (predictions_df.select(col(label_col).cast("int").alias("y"),
                                        vector_to_array(col(raw_pred_col))[pos_index].alias("s1")).cache())

        # materialize so timing/metrics reflect actual execution
        _ = scored.count()

        p, r, f1, tp, fp, fn = ModelEvaluation.prf_at_threshold_fast(scored, thr=thr, label_col="y", score_col="s1")
        return {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn}







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
