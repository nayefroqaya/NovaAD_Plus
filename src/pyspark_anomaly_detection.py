import time
import warnings

import colorama
import numpy as np
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.classification import LogisticRegression, GBTClassifier
# from pyspark.sql.functions import col, when, lit, vector_to_array, sum as spark_sum
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import (RandomForestClassifier, DecisionTreeClassifier, LogisticRegression)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, expr
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.sql.functions import col, when
from pyspark.sql.functions import col, when
from pyspark.sql.functions import col, when, lit, sum as spark_sum, udf
from pyspark.sql.functions import col, when, lit, udf, sum as spark_sum
from pyspark.sql.functions import lit
from pyspark.sql.functions import udf, col
# from pyspark.sql.functions import col, when, lit, vector_to_array, sum as spark_sum
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.types import ArrayType, DoubleType
from scipy.stats import randint, uniform
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sparkxgb import XGBoostClassifier
from pyspark.sql.functions import col, when, abs, log
from pyspark.sql.functions import col, log, when, abs as _abs
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.sql.functions import col, when
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import IntegerType

from pyspark.ml.classification import RandomForestClassifier
# If you want GBT instead, swap classifier block below.
# from pyspark.ml.classification import GBTClassifier

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:

    @staticmethod
    def anomaly_detector(df_final_train, df_test, mode):

        '''

        df_final_train = df_final_train.select("Node_block_id", "features_vec_final", "Final_Label")

        #df_val = df_val.select("Node_block_id", "features_vec_final", "Final_Label")
        df_test = df_test.select("Node_block_id", "features_vec_final", "Final_Label")

        train_df = (df_final_train.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                                 "features").withColumnRenamed(
            "Final_Label", "label").cache())

        #val_df = (df_val.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                       "features").withColumnRenamed(
            "Final_Label", "label").cache())

        test_df = (df_test.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                         "features").withColumnRenamed(
            "Final_Label", "label").cache())

        train_df.count()
        val_df.count()
        test_df.count()
        #exit()

        if mode == 'M':
            # =====================================================
            # 1. AUTOMATIC CLASS WEIGHTS (SAFE)
            # =====================================================
            label_counts = train_df.groupBy("label").count().collect()
            total_count = sum(r["count"] for r in label_counts)

            class_weights = {r["label"]: total_count / (2.0 * r["count"]) for r in label_counts}

            print("Class weights:", class_weights)

            train_df = train_df.withColumn("class_weight",
                when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(class_weights.get(1, 1.0)))

            val_df = val_df.withColumn("class_weight",
                when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(class_weights.get(1, 1.0)))

            test_df = test_df.withColumn("class_weight",
                when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(class_weights.get(1, 1.0)))

            print("TRAIN COLS:", train_df.columns)
            # -------------------------------
            # 3. Base models
            # -------------------------------
            # -----------------------------
            # 2. Define base models
            # -----------------------------
            lr_model = LogisticRegression(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="lr_prob", predictionCol="lr_pred", maxIter=150, regParam=0.01, elasticNetParam=0.0)

            rf_model = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="rf_prob", rawPredictionCol="rf_raw", predictionCol="rf_pred", numTrees=200, maxDepth=20,
                minInstancesPerNode=10, subsamplingRate=0.8, featureSubsetStrategy="sqrt")

            svc_model = LinearSVC(featuresCol="features", labelCol="label", weightCol="class_weight",
                predictionCol="svc_pred", rawPredictionCol="svc_raw", maxIter=100, regParam=0.01)

            # -----------------------------
            # 3. Train base models
            # -----------------------------
            start_fit = time.time()
            lr_model = lr_model.fit(train_df)
            rf_model = rf_model.fit(train_df)
            svc_model = svc_model.fit(train_df)
            end_fit = time.time()
            fit_time = (end_fit - start_fit) / 60

            # -----------------------------
            # 4. Prepare meta features
            # -----------------------------
            def add_probs(df, model, prob_col, prefix):
                return (
                    model.transform(df).withColumn(f"{prefix}_arr", vector_to_array(prob_col)).withColumn(f"{prefix}_0",
                                                                                                          col(f"{prefix}_arr")[
                                                                                                              0]).withColumn(
                        f"{prefix}_1", col(f"{prefix}_arr")[1]).withColumn(f"{prefix}_margin",
                                                                           abs(col(f"{prefix}_1") - col(
                                                                               f"{prefix}_0"))).withColumn(
                        f"{prefix}_entropy", -(
                                    col(f"{prefix}_1") * log(col(f"{prefix}_1") + 1e-9) + col(f"{prefix}_0") * log(
                                col(f"{prefix}_0") + 1e-9))))

            def prepare_meta(df):
                df = add_probs(df, lr_model, "lr_prob", "lr")
                df = add_probs(df, rf_model, "rf_prob", "rf")
                # LinearSVC output
                df = df.drop("rawPrediction") if "rawPrediction" in df.columns else df
                df = svc_model.transform(df).withColumn("svc_pred_val", col("svc_pred").cast("double"))
                return df

            train_meta = prepare_meta(train_df)
            val_meta = prepare_meta(val_df)
            test_meta = prepare_meta(test_df)

            # -----------------------------
            # 5. Assemble meta features
            # -----------------------------
            meta_features = ["lr_0", "lr_1", "lr_margin", "lr_entropy", "rf_0", "rf_1", "rf_margin", "rf_entropy",
                             "svc_pred_val"]

            assembler = VectorAssembler(inputCols=meta_features, outputCol="meta_features")
            train_meta = assembler.transform(train_meta)
            val_meta = assembler.transform(val_meta)
            test_meta = assembler.transform(test_meta)

            # -----------------------------
            # 6. Train meta LogisticRegression
            # -----------------------------
            meta_lr = LogisticRegression(featuresCol="meta_features", labelCol="label", weightCol="class_weight",
                predictionCol="last_pred_label", probabilityCol="final_prob", rawPredictionCol="meta_raw", maxIter=100,
                regParam=0.01, elasticNetParam=0.0)

            stack_model = meta_lr.fit(train_meta)

            # -----------------------------
            # 7. Threshold tuning on validation
            # -----------------------------
            val_preds = stack_model.transform(val_meta).withColumn("prob_1", vector_to_array("final_prob")[1])
            thresholds = [i / 100 for i in range(10, 90)]
            best_f1 = -1
            best_threshold = 0.5

            for t in thresholds:
                preds = val_preds.withColumn("pred_adj", when(col("prob_1") >= t, 1).otherwise(0))
                tp = preds.filter("label=1 AND pred_adj=1").count()
                fp = preds.filter("label=0 AND pred_adj=1").count()
                fn = preds.filter("label=1 AND pred_adj=0").count()
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

            print(f"🔥 Optimal threshold (validation F1): {best_threshold:.2f} | F1: {best_f1:.4f}")

            # -----------------------------
            # 8. Test predictions
            # -----------------------------
            start_predict = time.time()
            test_preds = stack_model.transform(test_meta).withColumn("prob_1", vector_to_array("final_prob")[1])
            final_test_predictions = test_preds.withColumn("last_pred_label",
                when(col("prob_1") >= best_threshold, 1).otherwise(0))
            end_predict = time.time()
            predict_time = (end_predict - start_predict) / 60

            # -----------------------------
            # 9. Return results
            # -----------------------------
            return {"predictions_df": final_test_predictions, "best_threshold": best_threshold, "fit_time": fit_time,
                "predict_time": predict_time}
        '''
        if mode=='X':   # ------------------------------------------------------------------------------------------------------

            # ============================================================
            # FULL COPY/PASTE BLOCK (NO pca_model REQUIRED)
            # - Trains a supervised classifier on df_final_train
            # - Handles class imbalance with weightCol
            # - Creates internal val split (if no validation set)
            # - Tunes probability threshold on internal val (F1)
            # - Predicts on df_test
            # - FIXES: "Vectors MUST NOT be Null" by filtering null feature vectors
            #
            # REQUIREMENTS:
            #   df_final_train: has Final_Label + ONE vector feature column
            #   df_test       : has Final_Label + same vector feature column
            #
            # It will auto-pick the first available feature col in:
            #   ["pca_features", "features_vec_final", "features"]
            # ============================================================

            from pyspark.sql import functions as F
            from pyspark.sql.functions import col, when, lit, udf
            from pyspark.sql.types import IntegerType

            from pyspark.ml.classification import RandomForestClassifier
            from sklearn.metrics import classification_report, f1_score

            # -----------------------------
            # 0) Auto-detect feature column
            # -----------------------------
            CANDIDATE_FEATURE_COLS = ["pca_features", "features_vec_final", "features"]
            feature_col = next(
                (c for c in CANDIDATE_FEATURE_COLS if (c in df_final_train.columns and c in df_test.columns)), None)

            if feature_col is None:
                raise ValueError(f"No common feature vector column found. Need one of {CANDIDATE_FEATURE_COLS} "
                                 f"in BOTH df_final_train and df_test.\n"
                                 f"df_final_train cols={df_final_train.columns}\n"
                                 f"df_test cols={df_test.columns}")

            print(f"[INFO] Using feature column: {feature_col}")

            # -----------------------------
            # 1) Build clean train/test (drop NULL vectors + NULL labels)
            # -----------------------------
            train_full = (df_final_train.select(col(feature_col).alias("features"),
                                                col("Final_Label").cast("int").alias("label")).filter(
                col("features").isNotNull()).filter(col("label").isNotNull()))

            test_df = (df_test.select(col(feature_col).alias("features"),
                                      col("Final_Label").cast("int").alias("label")).filter(
                col("features").isNotNull()).filter(col("label").isNotNull()))

            print(test_df.count())
            #exit()


            # Quick diagnostics
            orig_train_n = df_final_train.count()
            clean_train_n = train_full.count()
            orig_test_n = df_test.count()
            clean_test_n = test_df.count()

            print(f"[INFO] Train rows: {orig_train_n} -> {clean_train_n} after filtering NULL features/labels")
            print(f"[INFO] Test rows : {orig_test_n} -> {clean_test_n} after filtering NULL features/labels")

            if clean_train_n == 0:
                raise ValueError("No training rows left after filtering NULL feature vectors.")
            if clean_test_n == 0:
                raise ValueError("No test rows left after filtering NULL feature vectors.")

            # -----------------------------
            # 2) Internal train/val split (since you don't have validation data)
            # -----------------------------
            train_part, val_part = train_full.randomSplit([0.8, 0.2], seed=42)

            # Ensure both classes exist in train_part
            cnt_rows = train_part.groupBy("label").count().collect()
            cnt = {int(r["label"]): int(r["count"]) for r in cnt_rows}
            n0, n1 = cnt.get(0, 0), cnt.get(1, 0)
            print(f"[INFO] train_part class counts: normal(0)={n0}, anomaly(1)={n1}")

            if n0 == 0 or n1 == 0:
                # If this happens, use a different split seed or ratio
                raise ValueError("Internal split produced only one class. Try seed=1..100 or split=[0.9,0.1].")

            # -----------------------------
            # 3) Handle imbalance via weightCol (recommended)
            # -----------------------------
            w0 = (n0 + n1) / (2.0 * n0)
            w1 = (n0 + n1) / (2.0 * n1)

            print(f"[INFO] Weights -> w0={w0:.4f}, w1={w1:.4f}")

            train_part = train_part.withColumn("weight", when(col("label") == 1, lit(w1)).otherwise(lit(w0)))

            # -----------------------------
            # 4) Train classifier (RandomForest is robust to noisy pseudo-labels)
            # -----------------------------
            rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight",
                predictionCol="prediction", probabilityCol="probability", numTrees=300, maxDepth=12, seed=42)

            model = rf.fit(train_part)

            # -----------------------------
            # 5) Tune threshold on internal val (optimize F1)
            # -----------------------------
            val_pred = model.transform(val_part)

            val_pdf = val_pred.select("label", "probability").toPandas()
            val_probs = val_pdf["probability"].apply(lambda v: float(v[1])).values
            val_true = val_pdf["label"].astype(int).values

            best_thr, best_f1 = 0.5, -1.0
            for thr in [i / 100 for i in range(5, 96, 1)]:  # 0.05..0.95
                val_hat = (val_probs > thr).astype(int)
                f1 = f1_score(val_true, val_hat, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr

            print(f"[INFO] Best threshold (internal val) = {best_thr:.2f} (F1={best_f1:.3f})")

            # -----------------------------
            # 6) Predict on test with tuned threshold
            # -----------------------------
            test_pred = model.transform(test_df)

            @udf(IntegerType())
            def prob_to_label(prob, thr=float(best_thr)):
                return 1 if float(prob[1]) > thr else 0

            test_pred = test_pred.withColumn("pred_thr", prob_to_label(col("probability")))

            # -----------------------------
            # 7) Report on test (if df_test Final_Label is true label)
            # -----------------------------
            test_pdf2 = test_pred.select("label", "pred_thr").toPandas()
            print("\n==================== TEST REPORT (thresholded) ====================")
            print(classification_report(test_pdf2["label"].values, test_pdf2["pred_thr"].values, digits=3))

