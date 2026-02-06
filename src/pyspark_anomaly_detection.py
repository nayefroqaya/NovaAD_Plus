
import warnings
import colorama

from pyspark.sql.functions import col, when, lit, udf
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# ✅ Alias Spark ML classes to avoid ANY shadowing / UnboundLocalError
from pyspark.ml.classification import (
    LogisticRegression as SparkLogisticRegression,
    RandomForestClassifier as SparkRandomForestClassifier,
    GBTClassifier as SparkGBTClassifier,
)

warnings.filterwarnings("ignore")
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:

    @staticmethod
    def anomaly_detector(df_final_train_cls, df_test_cls, df_val_cls, mode):

        if mode == "M":

            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # -------- checks --------
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("df_train_quality", df_final_train_cls), ("df_val_cls", df_val_cls),
                             ("df_test_cls", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_final_train_cls.select(ID_COL, FEAT_COL, LABEL_COL).cache()
            df_val = df_val_cls.select(ID_COL, FEAT_COL, LABEL_COL).cache()
            df_test = df_test_cls.select(ID_COL, FEAT_COL, LABEL_COL).cache()

            # Guard: empty datasets
            if df_val.rdd.isEmpty():
                raise ValueError("df_val_cls is empty -> cannot validate weights/threshold.")
            if df_train.rdd.isEmpty():
                raise ValueError("df_train_quality is empty -> cannot train.")
            if df_test.rdd.isEmpty():
                raise ValueError("df_test_cls is empty -> cannot test.")

            print("Train label distribution:")
            df_train.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            evaluator_pr = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderPR")

            def tune_model(estimator, param_grid, train_df, evaluator, parallelism=4, train_ratio=0.8):
                tvs = TrainValidationSplit(estimator=estimator, estimatorParamMaps=param_grid, evaluator=evaluator,
                    trainRatio=train_ratio, parallelism=parallelism)
                return tvs.fit(train_df).bestModel

            # ----------------------------
            # 1) Fit/tune base models
            # ----------------------------
            lr = SparkLogisticRegression(featuresCol=FEAT_COL, labelCol=LABEL_COL, maxIter=50)
            lr_grid = (ParamGridBuilder().addGrid(lr.regParam, [1e-4, 1e-3, 1e-2]).addGrid(lr.elasticNetParam,
                                                                                           [0.0, 0.5, 1.0]).build())
            lr_model = tune_model(lr, lr_grid, df_train, evaluator_pr)

            rf = SparkRandomForestClassifier(featuresCol=FEAT_COL, labelCol=LABEL_COL, seed=42)
            rf_grid = (ParamGridBuilder().addGrid(rf.numTrees, [100, 200]).addGrid(rf.maxDepth, [5, 10]).build())
            rf_model = tune_model(rf, rf_grid, df_train, evaluator_pr)

            gbt = SparkGBTClassifier(featuresCol=FEAT_COL, labelCol=LABEL_COL, seed=42)
            gbt_grid = (ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5]).addGrid(gbt.maxIter, [30, 60]).build())
            gbt_model = tune_model(gbt, gbt_grid, df_train, evaluator_pr)

            print("[OK] Base models tuned and trained.")

            # ----------------------------
            # 2) Validate: weights by AUC-PR
            # ----------------------------
            def score_with_p1(model, df, p_col_name):
                return (
                    model.transform(df).select(ID_COL, LABEL_COL, vector_to_array("probability")[1].alias(p_col_name)))

            val_lr = score_with_p1(lr_model, df_val, "p_lr")
            val_rf = score_with_p1(rf_model, df_val, "p_rf").select(ID_COL, "p_rf")
            val_gbt = score_with_p1(gbt_model, df_val, "p_gbt").select(ID_COL, "p_gbt")

            # ✅ Join ONLY on ID_COL (avoid empty joins)
            val_scored = (val_lr.join(val_rf, ID_COL, "inner").join(val_gbt, ID_COL, "inner").cache())

            if val_scored.rdd.isEmpty():
                raise ValueError("val_scored is empty after joins. Check Node_block_id consistency/duplicates. "
                                 "Try ensuring unique IDs or using left joins.")

            @udf(VectorUDT())
            def to_raw_vec(p):
                p = float(p)
                return Vectors.dense([1.0 - p, p])

            weight_sets = [(0.34, 0.33, 0.33), (0.20, 0.20, 0.60), (0.20, 0.60, 0.20), (0.60, 0.20, 0.20),
                (1 / 3, 1 / 3, 1 / 3), ]

            best_w, best_aucpr = None, -1.0

            for w_lr, w_rf, w_gbt in weight_sets:
                tmp = (val_scored.withColumn("p_ens",
                                             w_lr * col("p_lr") + w_rf * col("p_rf") + w_gbt * col("p_gbt")).withColumn(
                    "rawPrediction", to_raw_vec(col("p_ens"))))
                # Guard: evaluator needs both classes present; still safe, but avoid empty
                if tmp.rdd.isEmpty():
                    continue
                aucpr = evaluator_pr.evaluate(tmp)
                if aucpr > best_aucpr:
                    best_aucpr = aucpr
                    best_w = (w_lr, w_rf, w_gbt)

            if best_w is None:
                raise ValueError(
                    "Could not compute AUC-PR on validation. Check that VAL contains both classes (0 and 1).")

            W_LR, W_RF, W_GBT = best_w
            print(f"[VAL] Best weights (LR, RF, GBT) = {best_w} | AUC-PR = {best_aucpr:.6f}")

            val_ens = val_scored.withColumn("p_ens",
                W_LR * col("p_lr") + W_RF * col("p_rf") + W_GBT * col("p_gbt")).cache()

            # ----------------------------
            # 3) Validate: threshold by best F1
            # ----------------------------
            def f1_at_threshold(df_with_p, thr):
                pred = df_with_p.select(col(LABEL_COL).cast("double").alias("label"),
                    when(col("p_ens") >= lit(thr), 1.0).otherwise(0.0).alias("prediction"))
                rdd = pred.rdd.map(lambda r: (r["prediction"], r["label"]))
                return MulticlassMetrics(rdd).fMeasure(1.0)

            thresholds = [i / 100 for i in range(1, 100)]
            best_thr, best_f1 = None, -1.0
            for t in thresholds:
                f1 = f1_at_threshold(val_ens, t)
                if f1 > best_f1:
                    best_f1, best_thr = f1, t

            print(f"[VAL] Best threshold = {best_thr:.2f} | F1 = {best_f1:.6f}")

            # ----------------------------
            # 4) Test: Precision/Recall/F1
            # ----------------------------
            test_lr = score_with_p1(lr_model, df_test, "p_lr")
            test_rf = score_with_p1(rf_model, df_test, "p_rf").select(ID_COL, "p_rf")
            test_gbt = score_with_p1(gbt_model, df_test, "p_gbt").select(ID_COL, "p_gbt")

            test_scored = test_lr.join(test_rf, ID_COL, "inner").join(test_gbt, ID_COL, "inner").cache()
            if test_scored.rdd.isEmpty():
                raise ValueError("test_scored is empty after joins. Check Node_block_id consistency in test.")

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

