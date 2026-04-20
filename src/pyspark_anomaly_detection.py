
import warnings
import colorama
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
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

from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import precision_recall_curve
from pyspark.sql.functions import when, lower, trim, col

import time
import numpy as np
from pyspark.sql.functions import col, lit, when, pmod
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

from pyspark.sql.functions import hash as ps_hash, abs as ps_abs

import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when, lower, trim, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import classification_report

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit, lower, trim, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import PCA as SparkPCA, StandardScaler
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.stat import Summarizer
import numpy as np
from sklearn.metrics import classification_report
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import classification_report, f1_score
import numpy as np

from pyspark.sql.functions import col, when, lit
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import classification_report
import pandas as pd

from pyspark.sql.functions import col, when, lit
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from pyspark.sql.functions import when

from pyspark.sql.functions import col, lit, explode, array_repeat
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import f1_score, classification_report
import numpy as np
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, lower, trim, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StandardScaler, PCA as SparkPCA, VectorAssembler
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.classification import GBTClassifier
from sklearn.metrics import classification_report, f1_score
import numpy as np
import time
from pyspark.sql.functions import col, lit, when, abs as ps_abs, hash as ps_hash, pmod
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import FMClassifier

import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings("ignore")
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:


    @staticmethod
    def anomaly_detector(df_full_train_labeled_features, sequences_df):
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql import functions as F

        # ------- classification stage. GBTClassifier-----New / Try ------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------
        # --------------------------
        # 0) Build training data
        # --------------------------

        # if DATASET=='BGL' or DATASET=='TH_1G' :
        # --------------------------
        # 0) Train data (real normal + ALL pseudo)
        # --------------------------case1 --------------------------------------------------------------------------
        df_real_normal = sequences_df.filter(col("Temp_label") == 0).select(col("Node_block_id"),
                                                                            col("features_vec_final"),
                                                                            lit(0).alias("Final_Label"),
                                                                            lit("real").alias("src"))

        df_pseudo_all = df_full_train_labeled_features.select(col("Node_block_id"), col("features_vec_final"),
                                                              col("Final_Label").cast("int").alias("Final_Label"),
                                                              lit("pseudo").alias("src"))

        train_df = df_real_normal.unionByName(df_pseudo_all)

        # --------------------------
        # 0.1) Deterministic split (NO rand, NO randomSplit)
        # split_key in [0..99] is stable for each Node_block_id every run
        # --------------------------
        train_df = train_df.withColumn("split_key", pmod(ps_abs(ps_hash(col("Node_block_id"))), lit(100)))

        train_base_df = train_df.filter(col("split_key") < 90).drop("split_key")
        val_df = train_df.filter(col("split_key") >= 90).drop("split_key")

        # --------------------------
        # 0.2) Weighting (cap weights to reduce swings)
        # --------------------------
        PSEUDO_TRUST = 0.6
       #** WEIGHT_CAP = 8.0  # smaller cap = more stable, fewer crazy shifts
        WEIGHT_CAP = 8.0 #4.0
        n0 = train_df.filter(col("Final_Label") == 0).count()
        n1 = train_df.filter(col("Final_Label") == 1).count()

        #**raw_w1 = float(n0 / max(n1, 1)) * 0.7
        raw_w1 = float(n0 / max(n1, 1)) * 0.7
        w1 = float(min(raw_w1, WEIGHT_CAP))
        w0 = 1.0

        print(
            f"[INFO] Train counts n0={n0}, n1={n1}, raw_w1={raw_w1:.4f}, capped_w1={w1:.4f}, PSEUDO_TRUST={PSEUDO_TRUST}")

        train_base_df = train_base_df.withColumn("baseClassWeight",
                                                 when(col("Final_Label") == 1, lit(w1)).otherwise(lit(w0))).withColumn(
            "srcWeight", when(col("src") == "pseudo", lit(PSEUDO_TRUST)).otherwise(lit(1.0))).withColumn("classWeight",
                                                                                                         col("baseClassWeight") * col(
                                                                                                             "srcWeight"))

        val_df = val_df.withColumn("baseClassWeight",
                                   when(col("Final_Label") == 1, lit(w1)).otherwise(lit(w0))).withColumn("srcWeight",
                                                                                                         when(
                                                                                                             col("src") == "pseudo",
                                                                                                             lit(PSEUDO_TRUST)).otherwise(
                                                                                                             lit(1.0))).withColumn(
            "classWeight", col("baseClassWeight") * col("srcWeight"))

        # --------------------------
        # 0.3) Test set
        # --------------------------
        test_df = sequences_df.filter(col("Temp_label") == 888).select(col("Node_block_id"), col("features_vec_final"),
                                                                       col("y_true").alias("Final_Label"),
                                                                       col("anomaly_score_pca") if "anomaly_score_pca" in sequences_df.columns else lit(
                                                                           0.0).alias("anomaly_score_pca"),
                                                                       col("anomaly_score_gmm") if "anomaly_score_gmm" in sequences_df.columns else lit(
                                                                           0.0).alias("anomaly_score_gmm"),
                                                                       col("pca_flag") if "pca_flag" in sequences_df.columns else lit(
                                                                           0).alias("pca_flag"),
                                                                       col("gmm_flag") if "gmm_flag" in sequences_df.columns else lit(
                                                                           0).alias("gmm_flag")).withColumn(
            "Final_Label", col("Final_Label").cast("int"))

        # --------------------------
        # 1) Assemble features
        # --------------------------
        feature_cols = ["features_vec_final"]
        if "anomaly_score_pca" in train_df.columns:
            feature_cols.append("anomaly_score_pca")
        if "anomaly_score_gmm" in train_df.columns:
            feature_cols.append("anomaly_score_gmm")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_augmented")

        train_base_df = assembler.transform(train_base_df)
        val_df = assembler.transform(val_df)
        test_df = assembler.transform(test_df)

        features_col = "features_augmented"

        # --------------------------
        # 1.1) CACHE + MATERIALIZE (critical for stability)
        # --------------------------
        # train_base_df = train_base_df.cache()
        train_base_df = train_base_df.persist(StorageLevel.MEMORY_AND_DISK)
        # val_df = val_df.cache()
        val_df = val_df.persist(StorageLevel.MEMORY_AND_DISK)
        # test_df = test_df.cache()
        test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)

        _ = train_base_df.count()
        _ = val_df.count()
        _ = test_df.count()

        # --------------------------
        # 2) Train deterministic GBT (reduce internal randomness)
        # --------------------------
        # maxIter=50, maxDepth=6 , stepSize=0.1
        # gbt = GBTClassifier(featuresCol=features_col, labelCol="Final_Label", weightCol="classWeight",
        #    maxIter=50, maxDepth=6, stepSize=0.1, seed=42, subsamplingRate= 1.0 ,  # 1.0
        #                    featureSubsetStrategy="all")

        from xgboost.spark import SparkXGBClassifier

        # gbt = SparkXGBClassifier(features_col=features_col, label_col="Final_Label", weight_col="classWeight",
        #    #num_workers=4,  # tune to your cluster
        #    max_depth=6, eta=0.1, n_estimators=200, subsample=1.0, colsample_bytree=1.0, seed=42)

        gbt = SparkXGBClassifier(features_col=features_col, label_col="Final_Label", weight_col="classWeight",
                                 max_depth=4, eta=0.05, n_estimators=500, subsample=0.85, colsample_bytree=0.80,
                                 scale_pos_weight=1.5
                                 , eval_metric="logloss", seed=42)



        # gbt = RandomForestClassifier(featuresCol=features_col, labelCol="Final_Label", weightCol="classWeight",
        #    numTrees=75, maxDepth=5, seed=SEED, subsamplingRate=0.9, featureSubsetStrategy="all")

        # gbt = FMClassifier(
        #            featuresCol=features_col,
        #            labelCol="Final_Label",
        #            stepSize=0.01,
        #            factorSize=8,     # dimension of factor vectors
        #            maxIter=75
        #        )



        t0 = time.time()
        model = gbt.fit(train_base_df)
        end_t0 = time.time()
        print(f"[INFO] case1 :  GBT fit time: {(end_t0 - t0) / 60:.2f} minutes")

        # --------------------------
        # 3) VAL: tune threshold
        # --------------------------
        val_pred = model.transform(val_df)
        val_pdf = val_pred.select(col("Final_Label").alias("y"),
                                  vector_to_array(col("probability")).getItem(1).alias("prob_1"),
                                  col("pca_flag") if "pca_flag" in val_pred.columns else lit(0).alias("pca_flag"),
                                  col("gmm_flag") if "gmm_flag" in val_pred.columns else lit(0).alias(
                                      "gmm_flag")).toPandas()

        y_val = val_pdf["y"].values.astype(int)
        p_val = val_pdf["prob_1"].values
        pca_v = val_pdf["pca_flag"].values.astype(int) if "pca_flag" in val_pdf.columns else np.zeros_like(y_val)
        gmm_v = val_pdf["gmm_flag"].values.astype(int) if "gmm_flag" in val_pdf.columns else np.zeros_like(y_val)

       #** TARGET_RECALL = 0.95  # 94 0.95
        TARGET_RECALL = 0.60 #0.80
        best_threshold, best_prec = 0.5, -1.0  # 0.5

        for t in np.arange(0.01, 0.999, 0.005):
            preds = (p_val >= t).astype(int)
            r = recall_score(y_val, preds, pos_label=1)
            if r >= TARGET_RECALL:
                p = precision_score(y_val, preds, pos_label=1, zero_division=0)
                if p > best_prec:
                    best_prec, best_threshold = p, float(t)



        if best_prec < 0:
            best_threshold, best_f1 = 0.5, -1.0
            for t in np.arange(0.01, 0.999, 0.005):
                preds = (p_val >= t).astype(int)
                f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_threshold = f1, float(t)
            print(f"[INFO] Threshold by best class-1 F1: t={best_threshold:.3f}, F1={best_f1:.4f}")
        else:
            print(
                f"[INFO] Threshold by precision@recall>= {TARGET_RECALL}: t={best_threshold:.3f}, precision={best_prec:.4f}")

        if (p_val >= best_threshold).sum() == 0:
            best_threshold = float(np.quantile(p_val, 0.90))
            print(f"[WARN] Threshold produced 0 anomalies on VAL. Using quantile fallback t={best_threshold:.6f}")

        # --------------------------
        # 3.1) VAL: auto-tune gate offset (deterministic)
        # --------------------------
        offset_grid = np.arange(0.06, 0.21, 0.02)  # stable, not too wide
        best_offset, best_f1_gate = 0.12, -1.0

        for off in offset_grid:
            gate_t = min(best_threshold + float(off), 0.999)
            gated_preds = ((p_val >= best_threshold) | ((p_val >= gate_t) & ((pca_v + gmm_v) >= 1))).astype(int)
            f1g = f1_score(y_val, gated_preds, pos_label=1, zero_division=0)
            if f1g > best_f1_gate:
                best_f1_gate, best_offset = f1g, float(off)

        print(f"[INFO] Best offset on VAL: {best_offset:.3f} (VAL class-1 F1={best_f1_gate:.4f})")

        # --------------------------
        # 4) TEST: gated ensemble using tuned offset
        # --------------------------
        t1 = time.time()
        test_pred = model.transform(test_df)
        end_t1 = time.time()
        print(f"[INFO] case1 : GBT predict time: {(end_t1 - t1) / 60:.2f} minutes")

        case1_test_pdf = test_pred.select(col("Final_Label").alias("y"),
                                          vector_to_array(col("probability")).getItem(1).alias("prob_1"),
                                          col("pca_flag"), col("gmm_flag")).toPandas()

        p_test = case1_test_pdf["prob_1"].values
        gate_t = min(best_threshold + best_offset, 0.999)

        case1_test_pdf["final_pred"] = ((p_test >= best_threshold) | ((p_test >= gate_t) & (
                (case1_test_pdf["pca_flag"].values + case1_test_pdf["gmm_flag"].values) >= 1))).astype(int)  # 1

        #case1_test_pdf["final_pred"] = (p_test >= best_threshold).astype(int)

        print("\n================Case1:  TEST CLASSIFICATION REPORT (HASH-split stable) ================")
        print(classification_report(case1_test_pdf["y"].astype(int), case1_test_pdf["final_pred"], digits=4))
        print(f"[INFO] best_threshold={best_threshold:.4f}, best_offset={best_offset:.3f}, gate_t={gate_t:.4f}")

        # --------------------------case2 --------------------------------------------------------------------------

        # else:
        # ------- classification stage. GBTClassifier-----New ------------------------------------------------------
        # ======================================
        # 0) Prepare training and test sets
        # ======================================
        df_train_normal = sequences_df.filter(col("Temp_label") == 0).select(col("Node_block_id"),
                                                                             col("features_vec_final"),
                                                                             lit(0).alias("Final_Label"))

        # Pseudo-labeled anomalies from novelty detection
        df_pseudo_anomalies = df_full_train_labeled_features.filter(col("Final_Label") == 1)

        # Merge normal + pseudo-labeled anomalies
        train_df = df_train_normal.unionByName(df_pseudo_anomalies)

        # Optional oversample anomalies
        n0 = train_df.filter(col("Final_Label") == 0).count()
        n1 = train_df.filter(col("Final_Label") == 1).count()
        k = int(np.ceil(n0 / max(n1, 1)))
        oversampled_anom = df_pseudo_anomalies
        for _ in range(k - 1):
            oversampled_anom = oversampled_anom.unionByName(df_pseudo_anomalies)
        train_df = train_df.unionByName(oversampled_anom)

        # Small validation split for threshold tuning (10% of train)
        train_base_df = train_df.sample(False, 0.9, seed=42)
        val_df = train_df.subtract(train_base_df)

        # Test set
        test_df = sequences_df.filter(col("Temp_label") == 888).select(col("Node_block_id"), col("features_vec_final"),
                                                                       col("y_true").alias("Final_Label"),
                                                                       col("anomaly_score_pca") if "anomaly_score_pca" in sequences_df.columns else lit(
                                                                           0.0).alias("anomaly_score_pca"),
                                                                       col("anomaly_score_gmm") if "anomaly_score_gmm" in sequences_df.columns else lit(
                                                                           0.0).alias("anomaly_score_gmm"),
                                                                       col("pca_flag") if "pca_flag" in sequences_df.columns else lit(
                                                                           0).alias("pca_flag"),
                                                                       col("gmm_flag") if "gmm_flag" in sequences_df.columns else lit(
                                                                           0).alias("gmm_flag")).withColumn(
            "Final_Label", col("Final_Label").cast("int"))

        # ======================================
        # 1) Assemble features
        # ======================================
        feature_cols = ["features_vec_final"]
        if "anomaly_score_pca" in train_df.columns:
            feature_cols.append("anomaly_score_pca")
        if "anomaly_score_gmm" in train_df.columns:
            feature_cols.append("anomaly_score_gmm")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_augmented")
        train_base_df = assembler.transform(train_base_df)
        val_df = assembler.transform(val_df)
        test_df = assembler.transform(test_df)
        features_col = "features_augmented"

        # ======================================
        # 2) Train GBT Classifier
        # ======================================
        # gbt = GBTClassifier(featuresCol=features_col, labelCol="Final_Label",
        #                    maxIter=75,  # 100
        #                    maxDepth=5, #6
        #    stepSize=0.1, # 0.05
        #                    seed=123)
        # maxIter=50, maxDepth=6 , stepSize=0.1

        # gbt = RandomForestClassifier(featuresCol=features_col, labelCol="Final_Label", numTrees=50, maxDepth=6,
        #    seed=42, subsamplingRate= 1.0,  #1.0
        #                             featureSubsetStrategy="all")

        gbt = SparkXGBClassifier(features_col=features_col, label_col="Final_Label", max_depth=4, eta=0.05,
                                 n_estimators=500, subsample=0.85, colsample_bytree=0.80, scale_pos_weight=1.5,
                                 eval_metric="logloss", seed=42)

        # gbt = FMClassifier(featuresCol=features_col, labelCol="Final_Label", stepSize=0.01,
        #                   factorSize=8,
        #    # dimension of factor vectors
        #    maxIter=75)

        # gbt = FMClassifier(
        #    featuresCol=features_col,
        #    labelCol="Final_Label",
        #    factorSize=32,
        #    regParam=0.001,
        #    stepSize=0.03,
        #    maxIter=150,
        #    miniBatchFraction=1.0,
        #    fitLinear=True,
        #    fitIntercept=True,
        #    solver="adamW",
        #    seed=42,                      # important
        #    probabilityCol="probability",
        #    rawPredictionCol="rawPrediction",
        #    predictionCol="prediction"
        # )

        start_fit_classification = time.time()
        model = gbt.fit(train_base_df)
        end_fit_classification = time.time()
        case2_Classification_time = (end_fit_classification - start_fit_classification) / 60
        print(f"final Model classification  completed in {case2_Classification_time:.2f} minutes")

        # ======================================
        # 3) Validation predictions and threshold tuning
        # ======================================

        # ================================
        # Validation predictions
        # ================================
        val_pred = model.transform(val_df)

        # Convert probability vector to array
        val_pdf = val_pred.select(col("Final_Label").alias("y"),
                                  vector_to_array(col("probability")).getItem(1).alias("prob_1")  # class 1 probability
                                  ).toPandas()

        # Threshold tuning
        best_threshold = 0.5
        best_f1 = -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            preds = (val_pdf["prob_1"].values >= t).astype(int)
            f1 = f1_score(val_pdf["y"].values.astype(int), preds, average="weighted")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        print("Best threshold from VAL:", best_threshold)
        print("Best VAL weighted F1:", best_f1)

        # ================================
        # Test predictions
        # ================================

        start_predict_classification = time.time()

        test_pred = model.transform(test_df)
        end_predict_classification = time.time()
        case2_Classification_pred_time = (end_predict_classification - start_predict_classification) / 60
        print(f"final Model predicts  completed in {case2_Classification_pred_time:.2f} minutes")

        case2_test_pdf = test_pred.select(col("Final_Label").alias("y"),
                                          vector_to_array(col("probability")).getItem(1).alias("prob_1"),
                                          # class 1 probability
                                          col("pca_flag"), col("gmm_flag")).toPandas()

        # Classifier predictions
        case2_test_pdf["pred_gbt"] = (case2_test_pdf["prob_1"] >= best_threshold).astype(int)

        # Optional ensemble: predict anomaly if >=1 signal
        case2_test_pdf["final_pred"] = (
                    (case2_test_pdf["pred_gbt"] + case2_test_pdf["pca_flag"] + case2_test_pdf["gmm_flag"]) >= 1).astype(
            int)

        case1_classification_time =(end_t0 - t0)
        case1_Classification_pred_time =(end_t1 - t1)
        return case1_test_pdf, case2_test_pdf, case1_classification_time, case1_Classification_pred_time, case2_Classification_time, case2_Classification_pred_time

        #spark.stop()
        #exit()









