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
from pyspark.sql.functions import col, when, lit
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.functions import vector_to_array
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import RandomForestClassifier
# If you want GBT instead, swap classifier block below.
# from pyspark.ml.classification import GBTClassifier
import pyspark.sql.functions as psf
from pyspark.sql.functions import col, when, lit
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:

    @staticmethod
    def anomaly_detector(df_train_quality,df_test_cls, df_val_cls, mode):


        if mode=='M':
            # ----------------------------
            # Columns
            # ----------------------------
            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # ----------------------------
            # Basic checks + caching
            # ----------------------------
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("df_train_quality", df_train_quality), ("df_val_cls", df_val_cls),
                             ("df_test_cls", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_train_quality.select(ID_COL, FEAT_COL, LABEL_COL).cache()
            df_val = df_val_cls.select(ID_COL, FEAT_COL, LABEL_COL).cache()
            df_test = df_test_cls.select(ID_COL, FEAT_COL, LABEL_COL).cache()

            print("Train label distribution:")
            df_train.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            # ----------------------------
            # Evaluator (good for anomalies)
            # ----------------------------
            evaluator_pr = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderPR")

            # ----------------------------
            # Fast tuner helper
            # ----------------------------
            def tune_model(estimator, param_grid, train_df, evaluator, parallelism=4, train_ratio=0.8):
                tvs = TrainValidationSplit(estimator=estimator, estimatorParamMaps=param_grid, evaluator=evaluator,
                    trainRatio=train_ratio, parallelism=parallelism)
                tvs_model = tvs.fit(train_df)
                return tvs_model.bestModel

            # ============================================================
            # 1) Train + Tune Base Models
            # ============================================================

            # ---- Logistic Regression ----
            lr = LogisticRegression(featuresCol=FEAT_COL, labelCol=LABEL_COL, maxIter=50)
            lr_grid = (ParamGridBuilder().addGrid(lr.regParam, [1e-4, 1e-3, 1e-2]).addGrid(lr.elasticNetParam,
                                                                                           [0.0, 0.5, 1.0]).build())
            lr_model = tune_model(lr, lr_grid, df_train, evaluator_pr, parallelism=4)

            # ---- Random Forest ----
            rf = RandomForestClassifier(featuresCol=FEAT_COL, labelCol=LABEL_COL, seed=42)
            rf_grid = (ParamGridBuilder().addGrid(rf.numTrees, [100, 200]).addGrid(rf.maxDepth, [5, 10]).build())
            rf_model = tune_model(rf, rf_grid, df_train, evaluator_pr, parallelism=4)

            # ---- Gradient Boosted Trees ----
            gbt = GBTClassifier(featuresCol=FEAT_COL, labelCol=LABEL_COL, seed=42)
            gbt_grid = (ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5]).addGrid(gbt.maxIter, [30, 60]).build())
            gbt_model = tune_model(gbt, gbt_grid, df_train, evaluator_pr, parallelism=4)

            print("[OK] Base models tuned and trained.")

            # ============================================================
            # 2) Validate: choose ensemble weights by AUC-PR
            # ============================================================

            def score_with_p1(model, df, p_col_name):
                # probability is Vector [p0, p1] => take p1
                return model.transform(df).select(ID_COL, LABEL_COL,
                                                  vector_to_array("probability")[1].alias(p_col_name))

            val_lr = score_with_p1(lr_model, df_val, "p_lr").select(ID_COL, LABEL_COL, "p_lr")
            val_rf = score_with_p1(rf_model, df_val, "p_rf").select(ID_COL, LABEL_COL, "p_rf")
            val_gbt = score_with_p1(gbt_model, df_val, "p_gbt").select(ID_COL, LABEL_COL, "p_gbt")

            val_scored = val_lr.join(val_rf, [ID_COL, LABEL_COL]).join(val_gbt, [ID_COL, LABEL_COL]).cache()

            @udf(VectorUDT())
            def to_raw_vec(p):
                p = float(p)
                return Vectors.dense([1.0 - p, p])

            weight_sets = [(0.34, 0.33, 0.33),  # (LR, RF, GBT)
                (0.20, 0.20, 0.60), (0.20, 0.60, 0.20), (0.60, 0.20, 0.20), (1 / 3, 1 / 3, 1 / 3), ]

            best_w = None
            best_aucpr = -1.0

            for w_lr, w_rf, w_gbt in weight_sets:
                tmp = (val_scored.withColumn("p_ens",
                                             w_lr * col("p_lr") + w_rf * col("p_rf") + w_gbt * col("p_gbt")).withColumn(
                    "rawPrediction", to_raw_vec(col("p_ens"))))
                aucpr = evaluator_pr.evaluate(tmp)
                if aucpr > best_aucpr:
                    best_aucpr = aucpr
                    best_w = (w_lr, w_rf, w_gbt)

            W_LR, W_RF, W_GBT = best_w
            print(f"[VAL] Best weights (LR, RF, GBT) = {best_w} | AUC-PR = {best_aucpr:.6f}")

            val_ens = val_scored.withColumn("p_ens",
                                            W_LR * col("p_lr") + W_RF * col("p_rf") + W_GBT * col("p_gbt")).cache()

            # ============================================================
            # 3) Validate: choose threshold by best F1 on VAL
            # ============================================================

            def f1_at_threshold(df_with_p, thr):
                pred = df_with_p.select(col(LABEL_COL).cast("double").alias("label"),
                    when(col("p_ens") >= lit(thr), 1.0).otherwise(0.0).alias("prediction"))
                rdd = pred.rdd.map(lambda r: (r["prediction"], r["label"]))
                return MulticlassMetrics(rdd).fMeasure(1.0)

            thresholds = [i / 100 for i in range(1, 100)]  # 0.01..0.99
            best_thr, best_f1 = None, -1.0
            for t in thresholds:
                f1 = f1_at_threshold(val_ens, t)
                if f1 > best_f1:
                    best_f1, best_thr = f1, t

            print(f"[VAL] Best threshold = {best_thr:.2f} | F1 = {best_f1:.6f}")

            # ============================================================
            # 4) Test: compute Precision / Recall / F1 on TEST
            # ============================================================

            test_lr = score_with_p1(lr_model, df_test, "p_lr").select(ID_COL, LABEL_COL, "p_lr")
            test_rf = score_with_p1(rf_model, df_test, "p_rf").select(ID_COL, LABEL_COL, "p_rf")
            test_gbt = score_with_p1(gbt_model, df_test, "p_gbt").select(ID_COL, LABEL_COL, "p_gbt")

            test_scored = test_lr.join(test_rf, [ID_COL, LABEL_COL]).join(test_gbt, [ID_COL, LABEL_COL])

            pred_test = (test_scored.withColumn("p_ens", W_LR * col("p_lr") + W_RF * col("p_rf") + W_GBT * col(
                "p_gbt")).withColumn("prediction", when(col("p_ens") >= lit(best_thr), 1).otherwise(0)).cache())

            rdd_test = pred_test.select(col("prediction").cast("double"), col(LABEL_COL).cast("double")).rdd.map(tuple)

            metrics = MulticlassMetrics(rdd_test)

            precision = metrics.precision(1.0)
            recall = metrics.recall(1.0)
            f1 = metrics.fMeasure(1.0)

            print("\n=== ENSEMBLE PERFORMANCE ON TEST ===")
            print(f"Precision (anomaly=1): {precision:.4f}")
            print(f"Recall    (anomaly=1): {recall:.4f}")
            print(f"F1-score  (anomaly=1): {f1:.4f}")

            print("\nConfusion Matrix (rows=pred, cols=true):")
            print(metrics.confusionMatrix())



        else:   # ------------------------------------------------------------------------------------------------------

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
                (c for c in CANDIDATE_FEATURE_COLS if (c in df_train_quality.columns and c in df_test_cls.columns)), None)

            if feature_col is None:
                raise ValueError(f"No common feature vector column found. Need one of {CANDIDATE_FEATURE_COLS} "
                                 f"in BOTH df_final_train and df_test.\n"
                                 f"df_final_train cols={df_train_quality.columns}\n"
                                 f"df_test cols={df_test_cls.columns}")

            print(f"[INFO] Using feature column: {feature_col}")

            # -----------------------------
            # 1) Build clean train/test (drop NULL vectors + NULL labels)
            # -----------------------------
            train_full = (df_train_quality.select(col(feature_col).alias("features"),
                                                col("Final_Label").cast("int").alias("label")).filter(
                col("features").isNotNull()).filter(col("label").isNotNull()))

            test_df = (df_test_cls.select(col(feature_col).alias("features"),
                                      col("Final_Label").cast("int").alias("label")).filter(
                col("features").isNotNull()).filter(col("label").isNotNull()))

            print(test_df.count())
            #exit()


            # Quick diagnostics
            orig_train_n = df_train_quality.count()
            clean_train_n = train_full.count()
            orig_test_n = df_test_cls.count()
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

