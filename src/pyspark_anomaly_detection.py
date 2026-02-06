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

            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # -----------------------------
            # 0) Safety: filter null vectors and keep minimal columns
            # -----------------------------
            train_df = df_train_quality.filter(psf.col(FEAT_COL).isNotNull()).select(ID_COL, FEAT_COL, LABEL_COL)
            val_df = df_val_cls.filter(psf.col(FEAT_COL).isNotNull()).select(ID_COL, FEAT_COL, LABEL_COL)
            test_df = df_test_cls.filter(psf.col(FEAT_COL).isNotNull()).select(ID_COL, FEAT_COL, LABEL_COL)

            # -----------------------------
            # 1) Imbalance weights (fast + very useful)
            # -----------------------------
            n_pos = train_df.filter(psf.col(LABEL_COL) == 1).count()
            n_neg = train_df.filter(psf.col(LABEL_COL) == 0).count()

            if n_pos == 0 or n_neg == 0:
                print("[WARNING] Only one class in training data. No weighting.")
                train_df_w = train_df.withColumn("classWeight", psf.lit(1.0))
            else:
                w_pos = float(n_neg) / float(n_pos)
                train_df_w = train_df.withColumn("classWeight",
                    psf.when(psf.col(LABEL_COL) == 1, psf.lit(w_pos)).otherwise(psf.lit(1.0)))

            # -----------------------------
            # 2) Evaluator (AUC for tuning; stable)
            # -----------------------------
            auc_eval = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            # -----------------------------
            # 3) FAST tuning: RandomForest (small grid)
            # -----------------------------
            rf = RandomForestClassifier(labelCol=LABEL_COL, featuresCol=FEAT_COL, weightCol="classWeight",
                probabilityCol="rf_prob", predictionCol="rf_pred", rawPredictionCol="rawPrediction")

            rf_grid = (ParamGridBuilder().addGrid(rf.numTrees, [50, 100]).addGrid(rf.maxDepth, [5, 10]).build())

            rf_tvs = TrainValidationSplit(estimator=rf, estimatorParamMaps=rf_grid, evaluator=auc_eval, trainRatio=0.8,
                parallelism=4)

            print("\n[TRAIN] Fitting RF (TrainValidationSplit)...")
            rf_model = rf_tvs.fit(train_df_w)

            # -----------------------------
            # 4) FAST tuning: LogisticRegression (tiny grid)
            # -----------------------------
            lr = LogisticRegression(labelCol=LABEL_COL, featuresCol=FEAT_COL, weightCol="classWeight", maxIter=50,
                regParam=0.01, elasticNetParam=0.0, probabilityCol="lr_prob", predictionCol="lr_pred",
                rawPredictionCol="lr_raw")

            lr_grid = (ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.01, 0.1]).addGrid(lr.elasticNetParam,
                                                                                         [0.0, 0.5]).build())

            lr_auc_eval = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="lr_raw",
                metricName="areaUnderROC")

            lr_tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=lr_grid, evaluator=lr_auc_eval,
                trainRatio=0.8, parallelism=4)

            print("\n[TRAIN] Fitting LR (TrainValidationSplit)...")
            lr_model = lr_tvs.fit(train_df_w)

            # -----------------------------
            # 5) Pick best model on validation AUC
            # -----------------------------
            rf_val = rf_model.bestModel.transform(val_df)
            lr_val = lr_model.bestModel.transform(val_df)

            rf_val_auc = auc_eval.evaluate(rf_val.select("rawPrediction", LABEL_COL))
            lr_val_auc = lr_auc_eval.evaluate(lr_val.select("lr_raw", LABEL_COL))

            print(f"\n[VAL] RF AUC = {rf_val_auc:.4f}")
            print(f"[VAL] LR AUC = {lr_val_auc:.4f}")

            best_single = "rf" if rf_val_auc >= lr_val_auc else "lr"
            print(f"[SELECTED SINGLE MODEL] {best_single.upper()}")

            # -----------------------------
            # 6) Helper: per-class precision/recall/F1 in Spark
            # -----------------------------
            def print_per_class_metrics(pred_df, label_col, pred_col, title):
                rdd = (pred_df.select(psf.col(pred_col).cast("double"), psf.col(label_col).cast("double")).rdd.map(
                    lambda r: (r[0], r[1])))
                m = MulticlassMetrics(rdd)

                print(f"\n=== {title} ===")
                for c in [0.0, 1.0]:
                    print(f"Class {int(c)}: precision={m.precision(c):.3f}  "
                          f"recall={m.recall(c):.3f}  f1={m.fMeasure(c, 1.0):.3f}")
                print(f"Overall accuracy: {m.accuracy:.3f}")

            # -----------------------------
            # 7) TEST evaluation for best single model
            # -----------------------------
            if best_single == "rf":
                test_pred = rf_model.bestModel.transform(test_df)
                test_auc = auc_eval.evaluate(test_pred.select("rawPrediction", LABEL_COL))
                print(f"\n[TEST] Best single = RF, AUC = {test_auc:.4f}")
                print_per_class_metrics(test_pred, LABEL_COL, "rf_pred", "TEST metrics (RF)")
                return rf_model.bestModel
            else:
                test_pred = lr_model.bestModel.transform(test_df)
                test_auc = lr_auc_eval.evaluate(test_pred.select("lr_raw", LABEL_COL))
                print(f"\n[TEST] Best single = LR, AUC = {test_auc:.4f}")
                print_per_class_metrics(test_pred, LABEL_COL, "lr_pred", "TEST metrics (LR)")
                return lr_model.bestModel



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

