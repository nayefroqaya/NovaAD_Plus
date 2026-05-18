import time
import warnings

import colorama
import numpy as np
import pandas as pd

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import (
    LogisticRegression as SparkLogisticRegression,
    RandomForestClassifier as SparkRandomForestClassifier,
    GBTClassifier as SparkGBTClassifier,
)
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import StandardScaler, PCA as SparkPCA, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.stat import Summarizer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when, lower, trim, udf
from pyspark.sql.functions import abs as ps_abs, hash as ps_hash, pmod
from pyspark.sql.types import DoubleType
from pyspark.storagelevel import StorageLevel

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve

warnings.filterwarnings("ignore")

colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:

    @staticmethod
    def anomaly_detector(df_full_train_labeled_features, sequences_df):

        from xgboost.spark import SparkXGBClassifier

        # =====================================================================================
        # CASE 1: real normal + all pseudo labels
        # =====================================================================================

        # --------------------------
        # 0) Train data
        # --------------------------
        df_real_normal = sequences_df.filter(col("Temp_label") == 0).select(
            col("Node_block_id"),
            col("features_vec_final"),
            lit(0).alias("Final_Label"),
            lit("real").alias("src")
        )

        df_pseudo_all = df_full_train_labeled_features.select(
            col("Node_block_id"),
            col("features_vec_final"),
            col("Final_Label").cast("int").alias("Final_Label"),
            lit("pseudo").alias("src")
        )

        train_df = df_real_normal.unionByName(df_pseudo_all)

        # --------------------------
        # 0.1) Deterministic split
        # --------------------------
        train_df = train_df.withColumn(
            "split_key",
            pmod(ps_abs(ps_hash(col("Node_block_id"))), lit(100))
        )

        train_base_df = train_df.filter(col("split_key") < 90).drop("split_key")
        val_df = train_df.filter(col("split_key") >= 90).drop("split_key")

        # --------------------------
        # 0.2) Weighting
        # --------------------------
        PSEUDO_TRUST = 0.4
        WEIGHT_CAP = 8.0

        n0 = train_df.filter(col("Final_Label") == 0).count()
        n1 = train_df.filter(col("Final_Label") == 1).count()

        raw_w1 = float(n0 / max(n1, 1)) * 0.7
        w1 = float(min(raw_w1, WEIGHT_CAP))
        w0 = 1.0

        print(
            f"[INFO] Train counts n0={n0}, n1={n1}, "
            f"raw_w1={raw_w1:.4f}, capped_w1={w1:.4f}, "
            f"PSEUDO_TRUST={PSEUDO_TRUST}"
        )

        train_base_df = train_base_df.withColumn(
            "baseClassWeight",
            when(col("Final_Label") == 1, lit(w1)).otherwise(lit(w0))
        ).withColumn(
            "srcWeight",
            when(col("src") == "pseudo", lit(PSEUDO_TRUST)).otherwise(lit(1.0))
        ).withColumn(
            "classWeight",
            col("baseClassWeight") * col("srcWeight")
        )

        val_df = val_df.withColumn(
            "baseClassWeight",
            when(col("Final_Label") == 1, lit(w1)).otherwise(lit(w0))
        ).withColumn(
            "srcWeight",
            when(col("src") == "pseudo", lit(PSEUDO_TRUST)).otherwise(lit(1.0))
        ).withColumn(
            "classWeight",
            col("baseClassWeight") * col("srcWeight")
        )

        # --------------------------
        # 0.3) Test set
        # --------------------------
        test_df = sequences_df.filter(col("Temp_label") == 888).select(
            col("Node_block_id"),
            col("features_vec_final"),
            col("y_true").alias("Final_Label")
        ).withColumn(
            "Final_Label",
            col("Final_Label").cast("int")
        )

        # --------------------------
        # 1) Assemble features
        # Use only features_vec_final.
        # Exclude anomaly_score_pca, anomaly_score_gmm, pca_flag, gmm_flag.
        # --------------------------
        feature_cols = ["features_vec_final"]

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_augmented"
        )

        train_base_df = assembler.transform(train_base_df)
        val_df = assembler.transform(val_df)
        test_df = assembler.transform(test_df)

        features_col = "features_augmented"

        # --------------------------
        # 1.1) Cache + materialize
        # --------------------------
        train_base_df = train_base_df.persist(StorageLevel.MEMORY_AND_DISK)
        val_df = val_df.persist(StorageLevel.MEMORY_AND_DISK)
        test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

        _ = train_base_df.count()
        _ = val_df.count()
        _ = test_df.count()

        # --------------------------
        # 2) Train XGB
        # --------------------------
        gbt_case1 = SparkXGBClassifier(
            features_col=features_col,
            label_col="Final_Label",
            weight_col="classWeight",
            max_depth=4,
            eta=0.05,
            n_estimators=500,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=1.5,
            num_workers=2,
            eval_metric="logloss",
            seed=42
        )

        t0 = time.time()
        model_case1 = gbt_case1.fit(train_base_df)
        end_t0 = time.time()

        print(f"[INFO] case1 : XGB fit time: {(end_t0 - t0) / 60:.2f} minutes")

        # --------------------------
        # 3) Validation threshold tuning
        # --------------------------
        val_pred = model_case1.transform(val_df)

        val_pdf = val_pred.select(
            col("Final_Label").alias("y"),
            vector_to_array(col("probability")).getItem(1).alias("prob_1")
        ).toPandas()

        y_val = val_pdf["y"].values.astype(int)
        p_val = val_pdf["prob_1"].values

        target_recall_grid = [0.60, 0.80, 0.90, 0.95]

        best_global_f1 = -1
        best_target_recall = None
        best_threshold = 0.5
        best_prec = -1

        for target_recall in target_recall_grid:
            for t in np.arange(0.01, 0.999, 0.005):
                preds = (p_val >= t).astype(int)

                r = recall_score(y_val, preds, pos_label=1, zero_division=0)
                p = precision_score(y_val, preds, pos_label=1, zero_division=0)
                f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)

                if r >= target_recall:
                    if f1 > best_global_f1:
                        best_global_f1 = f1
                        best_target_recall = target_recall
                        best_threshold = float(t)
                        best_prec = p

        print(
            f"[INFO] Auto TARGET_RECALL={best_target_recall}, "
            f"threshold={best_threshold:.3f}, "
            f"VAL precision={best_prec:.4f}, "
            f"VAL F1={best_global_f1:.4f}"
        )

        TARGET_RECALL = best_target_recall
        best_threshold, best_prec = 0.5, -1.0

        for t in np.arange(0.01, 0.999, 0.005):
            preds = (p_val >= t).astype(int)
            r = recall_score(y_val, preds, pos_label=1, zero_division=0)

            if r >= TARGET_RECALL:
                p = precision_score(y_val, preds, pos_label=1, zero_division=0)

                if p > best_prec:
                    best_prec = p
                    best_threshold = float(t)

        if best_prec < 0:
            best_threshold, best_f1 = 0.5, -1.0

            for t in np.arange(0.01, 0.999, 0.005):
                preds = (p_val >= t).astype(int)
                f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(t)

            print(
                f"[INFO] Threshold by best class-1 F1: "
                f"t={best_threshold:.3f}, F1={best_f1:.4f}"
            )

        else:
            print(
                f"[INFO] Threshold by precision@recall>= {TARGET_RECALL}: "
                f"t={best_threshold:.3f}, precision={best_prec:.4f}"
            )

        if (p_val >= best_threshold).sum() == 0:
            best_threshold = float(np.quantile(p_val, 0.90))
            print(
                f"[WARN] Threshold produced 0 anomalies on VAL. "
                f"Using quantile fallback t={best_threshold:.6f}"
            )

        # --------------------------
        # 4) Test prediction
        # Classifier-only final decision.
        # --------------------------
        t1 = time.time()
        test_pred = model_case1.transform(test_df)
        end_t1 = time.time()

        print(f"[INFO] case1 : XGB predict time: {(end_t1 - t1) / 60:.2f} minutes")

        case1_test_pdf = test_pred.select(
            col("Final_Label").alias("y"),
            vector_to_array(col("probability")).getItem(1).alias("prob_1")
        ).toPandas()

        p_test = case1_test_pdf["prob_1"].values

        case1_test_pdf["final_pred"] = (p_test >= best_threshold).astype(int)

        print("\n================Case1: TEST CLASSIFICATION REPORT (HASH-split stable) ================")
        print(
            classification_report(
                case1_test_pdf["y"].astype(int),
                case1_test_pdf["final_pred"],
                digits=4
            )
        )
        print(f"[INFO] best_threshold={best_threshold:.4f}")

        # =====================================================================================
        # CASE 2: real normal + pseudo anomalies
        # =====================================================================================

        # --------------------------
        # 0) Prepare training and test sets
        # --------------------------
        df_train_normal = sequences_df.filter(col("Temp_label") == 0).select(
            col("Node_block_id"),
            col("features_vec_final"),
            lit(0).alias("Final_Label")
        )

        df_pseudo_anomalies = df_full_train_labeled_features.filter(
            col("Final_Label") == 1
        ).select(
            col("Node_block_id"),
            col("features_vec_final"),
            col("Final_Label").cast("int").alias("Final_Label")
        )

        train_df = df_train_normal.unionByName(df_pseudo_anomalies)

        # --------------------------
        # Optional oversampling
        # --------------------------
        n0 = train_df.filter(col("Final_Label") == 0).count()
        n1 = train_df.filter(col("Final_Label") == 1).count()

        k = int(np.ceil(n0 / max(n1, 1)))

        oversampled_anom = df_pseudo_anomalies

        for _ in range(k - 1):
            oversampled_anom = oversampled_anom.unionByName(df_pseudo_anomalies)

        train_df = train_df.unionByName(oversampled_anom)

        # --------------------------
        # Validation split
        # --------------------------
        train_base_df = train_df.sample(False, 0.9, seed=42)
        val_df = train_df.subtract(train_base_df)

        # --------------------------
        # Test set
        # --------------------------
        test_df = sequences_df.filter(col("Temp_label") == 888).select(
            col("Node_block_id"),
            col("features_vec_final"),
            col("y_true").alias("Final_Label")
        ).withColumn(
            "Final_Label",
            col("Final_Label").cast("int")
        )

        # --------------------------
        # 1) Assemble features
        # Use only features_vec_final.
        # --------------------------
        feature_cols = ["features_vec_final"]

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_augmented"
        )

        train_base_df = assembler.transform(train_base_df)
        val_df = assembler.transform(val_df)
        test_df = assembler.transform(test_df)

        features_col = "features_augmented"

        # --------------------------
        # 2) Train XGB Classifier
        # --------------------------
        gbt_case2 = SparkXGBClassifier(
            features_col=features_col,
            label_col="Final_Label",
            max_depth=4,
            eta=0.05,
            n_estimators=500,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=1.5,
            num_workers=2,
            eval_metric="logloss",
            seed=42
        )

        start_fit_classification = time.time()
        model_case2 = gbt_case2.fit(train_base_df)
        end_fit_classification = time.time()

        case2_Classification_time = (
            end_fit_classification - start_fit_classification
        ) / 60

        print(
            f"final Model classification completed in "
            f"{case2_Classification_time:.2f} minutes"
        )

        # --------------------------
        # 3) Validation predictions and threshold tuning
        # --------------------------
        val_pred = model_case2.transform(val_df)

        val_pdf = val_pred.select(
            col("Final_Label").alias("y"),
            vector_to_array(col("probability")).getItem(1).alias("prob_1")
        ).toPandas()

        best_threshold_case2 = 0.5
        best_f1 = -1.0

        for t in np.arange(0.05, 0.96, 0.01):
            preds = (val_pdf["prob_1"].values >= t).astype(int)

            f1 = f1_score(
                val_pdf["y"].values.astype(int),
                preds,
                average="weighted"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold_case2 = float(t)

        print("Best threshold from VAL:", best_threshold_case2)
        print("Best VAL weighted F1:", best_f1)

        # --------------------------
        # 4) Test predictions
        # Classifier-only final decision.
        # --------------------------
        start_predict_classification = time.time()
        test_pred = model_case2.transform(test_df)
        end_predict_classification = time.time()

        case2_Classification_pred_time = (
            end_predict_classification - start_predict_classification
        ) / 60

        print(
            f"final Model predicts completed in "
            f"{case2_Classification_pred_time:.2f} minutes"
        )

        case2_test_pdf = test_pred.select(
            col("Final_Label").alias("y"),
            vector_to_array(col("probability")).getItem(1).alias("prob_1")
        ).toPandas()

        case2_test_pdf["final_pred"] = (
            case2_test_pdf["prob_1"] >= best_threshold_case2
        ).astype(int)

        print("\n================Case2: TEST CLASSIFICATION REPORT ================")
        print(
            classification_report(
                case2_test_pdf["y"].astype(int),
                case2_test_pdf["final_pred"],
                digits=4
            )
        )

        case1_classification_time = end_t0 - t0
        case1_Classification_pred_time = end_t1 - t1

        return (
            best_target_recall,
            model_case1,
            model_case2,
            case1_test_pdf,
            case2_test_pdf,
            case1_classification_time,
            case1_Classification_pred_time,
            case2_Classification_time,
            case2_Classification_pred_time
        )