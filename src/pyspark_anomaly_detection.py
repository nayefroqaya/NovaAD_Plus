
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
        from pyspark.ml.functions import vector_to_array
        from pyspark.sql import functions as F

        if mode == "M":

            # ------------------------------------------------------------
            # Columns
            # ------------------------------------------------------------
            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # ------------------------------------------------------------
            # Silence Spark plans
            # ------------------------------------------------------------
            spark.sparkContext.setLogLevel("WARN")

            # ------------------------------------------------------------
            # Basic checks
            # ------------------------------------------------------------
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("TRAIN", df_final_train_cls), ("VAL", df_val_cls), ("TEST", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_final_train_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            df_val = df_val_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            df_test = df_test_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            # ------------------------------------------------------------
            # Class weights (important for SOMA)
            # ------------------------------------------------------------
            n_pos = df_train.filter(col(LABEL_COL) == 1).count()
            n_neg = df_train.filter(col(LABEL_COL) == 0).count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Training must contain both classes")

            pos_w = min(10.0, float(n_neg) / float(n_pos))
            print(f"[INFO] pos_w = {pos_w:.3f}")

            df_train_w = df_train.withColumn("class_weight",
                when(col(LABEL_COL) == 1, lit(pos_w)).otherwise(lit(1.0))).cache()

            # ------------------------------------------------------------
            # Train Linear SVM
            # ------------------------------------------------------------
            svm = LinearSVC(featuresCol=FEAT_COL, labelCol=LABEL_COL, weightCol="class_weight", maxIter=120)

            param_grid = (ParamGridBuilder().addGrid(svm.regParam, [1e-5, 1e-4, 1e-3, 1e-2]).build())

            evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            tvs = TrainValidationSplit(estimator=svm, estimatorParamMaps=param_grid, evaluator=evaluator,
                trainRatio=0.8, parallelism=4)

            svm_model = tvs.fit(df_train_w).bestModel
            print("[OK] LinearSVC trained.")

            # ------------------------------------------------------------
            # Score VAL / TEST
            # ------------------------------------------------------------
            val_scored = (svm_model.transform(df_val).select(col(LABEL_COL).alias("y"),
                vector_to_array(col("rawPrediction"))[1].alias("score")).cache())

            test_scored = (svm_model.transform(df_test).select(col(LABEL_COL).alias("y"),
                vector_to_array(col("rawPrediction"))[1].alias("score")).cache())

            # ------------------------------------------------------------
            # Fast precision / recall / F1 at threshold
            # ------------------------------------------------------------
            def prf(df, thr):
                tmp = df.select(col("y"), when(col("score") >= lit(thr), 1).otherwise(0).alias("yhat"))
                agg = tmp.agg(F.sum(((col("yhat") == 1) & (col("y") == 1)).cast("int")).alias("tp"),
                    F.sum(((col("yhat") == 1) & (col("y") == 0)).cast("int")).alias("fp"),
                    F.sum(((col("yhat") == 0) & (col("y") == 1)).cast("int")).alias("fn")).collect()[0]

                tp, fp, fn = int(agg.tp), int(agg.fp), int(agg.fn)
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r / (p + r + 1e-9)
                return p, r, f1

            # ------------------------------------------------------------
            # THRESHOLD SELECTION — F-beta (beta = 0.5)
            # ------------------------------------------------------------
            BETA = 0.5  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< KEY PARAMETER
            b2 = BETA * BETA

            qs = [i / 500 for i in range(1, 500)]  # 0.002 .. 0.998
            candidates = val_scored.approxQuantile("score", qs, 0.001)

            best_thr = None
            best_fbeta = -1
            best_p = best_r = best_f1 = None

            for t in candidates:
                p, r, f1 = prf(val_scored, float(t))
                fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-9)
                if fbeta > best_fbeta:
                    best_fbeta = fbeta
                    best_thr = float(t)
                    best_p, best_r, best_f1 = p, r, f1

            print(f"[VAL] Best threshold (F{BETA}) = {best_thr:.6f} | "
                  f"P={best_p:.4f} R={best_r:.4f} F1={best_f1:.4f}")

            # ------------------------------------------------------------
            # TEST METRICS
            # ------------------------------------------------------------
            p_test, r_test, f1_test = prf(test_scored, best_thr)

            print("\n=== TEST METRICS (LinearSVC, F0.5 optimized) ===")
            print(f"Precision (anomaly=1): {p_test:.4f}")
            print(f"Recall    (anomaly=1): {r_test:.4f}")
            print(f"F1-score  (anomaly=1): {f1_test:.4f}")

            exit()





            # -----------------------------
            # Columns
            # -----------------------------
            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # -----------------------------
            # Basic checks
            # -----------------------------
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("train", df_final_train_cls), ("val", df_val_cls), ("test", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_final_train_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_val = df_val_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_test = df_test_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            if df_train.rdd.isEmpty():
                raise ValueError("df_train is empty.")
            if df_val.rdd.isEmpty():
                raise ValueError("df_val is empty.")
            if df_test.rdd.isEmpty():
                raise ValueError("df_test is empty.")

            print("\n[INFO] TRAIN label distribution:")
            df_train.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            print("[INFO] VAL label distribution:")
            df_val.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            print("[INFO] TEST label distribution:")
            df_test.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            # -----------------------------
            # Class weights
            # -----------------------------
            n_pos = df_train.filter(col(LABEL_COL) == 1).count()
            n_neg = df_train.filter(col(LABEL_COL) == 0).count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Train must contain both classes 0 and 1.")

            pos_w = min(10.0, float(n_neg) / float(n_pos))  # allow up to 10 for hard sets
            df_train_w = df_train.withColumn("class_weight",
                when(col(LABEL_COL) == 1, lit(pos_w)).otherwise(lit(1.0))).cache()

            print(f"[INFO] pos_w = {pos_w:.3f}")

            # -----------------------------
            # TrainValidationSplit for LinearSVC
            # -----------------------------
            svm = LinearSVC(featuresCol=FEAT_COL, labelCol=LABEL_COL, weightCol="class_weight", maxIter=120)
            grid = (ParamGridBuilder().addGrid(svm.regParam, [1e-5, 1e-4, 1e-3, 1e-2]).build())

            # AUC-ROC is okay for tuning score model; thresholding handled separately
            evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            tvs = TrainValidationSplit(estimator=svm, estimatorParamMaps=grid, evaluator=evaluator, trainRatio=0.8,
                parallelism=4)

            svm_model = tvs.fit(df_train_w).bestModel
            print("[OK] Trained LinearSVC (best regParam).")

            # -----------------------------
            # Score VAL / TEST (score for class 1)
            # -----------------------------
            val_scored = (svm_model.transform(df_val).select(col(LABEL_COL).cast("int").alias("y"),
                vector_to_array(col("rawPrediction"))[1].alias("s1")).cache())

            test_scored = (svm_model.transform(df_test).select(col(LABEL_COL).cast("int").alias("y"),
                vector_to_array(col("rawPrediction"))[1].alias("s1")).cache())

            # -----------------------------
            # Fast PRF computation at threshold
            # -----------------------------
            def prf_at_threshold_fast(df, thr):
                tmp = df.select(col("y").alias("y"), when(col("s1") >= lit(thr), 1).otherwise(0).alias("yhat"))
                agg = tmp.agg(F.sum(((col("yhat") == 1) & (col("y") == 1)).cast("int")).alias("tp"),
                    F.sum(((col("yhat") == 1) & (col("y") == 0)).cast("int")).alias("fp"),
                    F.sum(((col("yhat") == 0) & (col("y") == 1)).cast("int")).alias("fn"),
                    F.sum(((col("yhat") == 0) & (col("y") == 0)).cast("int")).alias("tn"), ).collect()[0]

                tp, fp, fn, tn = int(agg["tp"]), int(agg["fp"]), int(agg["fn"]), int(agg["tn"])
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r / (p + r + 1e-9)
                return p, r, f1, tp, fp, fn, tn

            # ============================================================
            # THRESHOLD STRATEGY A: ALERT-BUDGET (Top-K rate)
            # Try several budgets and pick best by VAL F1 (or F-beta)
            # ============================================================
            # Budgets = maximum fraction of samples you are willing to alert as anomalies
            # Tight budget -> higher precision, lower recall
            BUDGETS = [0.05, 0.03, 0.02, 0.01, 0.005]  # adjust if needed

            def thr_from_budget(df_scored, budget):
                # Choose thr such that predicted positive rate approx = budget
                # => thr = quantile at (1 - budget)
                q = max(0.0, min(1.0, 1.0 - float(budget)))
                return float(df_scored.approxQuantile("s1", [q], 0.001)[0])

            val_candidates = []
            for b in BUDGETS:
                thr = thr_from_budget(val_scored, b)
                p, r, f1, *_ = prf_at_threshold_fast(val_scored, thr)
                val_candidates.append(("budget", b, thr, p, r, f1))

            # ============================================================
            # THRESHOLD STRATEGY B: Constraint-based (precision >= P_MIN)
            # Choose max recall subject to precision constraint
            # ============================================================
            P_MIN = 0.90
            qs = [i / 200 for i in range(1, 200)]  # 0.005..0.995
            cand_thr = val_scored.approxQuantile("s1", qs, 0.001)

            best_thr_c = None
            best_r_c = -1.0
            best_p_c = best_f1_c = None

            for t in cand_thr:
                p, r, f1, *_ = prf_at_threshold_fast(val_scored, float(t))
                if p >= P_MIN and r > best_r_c:
                    best_thr_c = float(t)
                    best_r_c = r
                    best_p_c = p
                    best_f1_c = f1

            if best_thr_c is not None:
                val_candidates.append(("constraint", P_MIN, best_thr_c, best_p_c, best_r_c, best_f1_c))

            # -----------------------------
            # Pick BEST candidate by VAL F1
            # -----------------------------
            val_candidates_sorted = sorted(val_candidates, key=lambda x: x[5], reverse=True)

            print("\n[VAL] Candidates (top 10 by F1):")
            print("rank | type       | param   | thr       | P      | R      | F1")
            for i, (typ, param, thr, p, r, f1) in enumerate(val_candidates_sorted[:10], 1):
                print(f"{i:>4} | {typ:<10} | {param:<6} | {thr:>8.5f} | {p:>6.3f} | {r:>6.3f} | {f1:>6.3f}")

            best_type, best_param, best_thr, p_val, r_val, f1_val = val_candidates_sorted[0]
            print(
                f"\n[VAL] SELECTED: {best_type}({best_param}) thr={best_thr:.6f} | P={p_val:.4f} R={r_val:.4f} F1={f1_val:.4f}")

            # -----------------------------
            # Evaluate TEST at chosen threshold
            # -----------------------------
            p_test, r_test, f1_test, tp, fp, fn, tn = prf_at_threshold_fast(test_scored, best_thr)

            print("\n=== TEST METRICS (LinearSVC) ===")
            print(f"Precision (anomaly=1): {p_test:.4f}")
            print(f"Recall    (anomaly=1): {r_test:.4f}")
            print(f"F1-score  (anomaly=1): {f1_test:.4f}")
            print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

            results = {"model": svm_model, "pos_w": float(pos_w), "chosen": {"type": best_type,
                                                                             "param": float(best_param) if isinstance(
                                                                                 best_param,
                                                                                 (int, float)) else best_param,
                                                                             "thr": float(best_thr),
                                                                             "val_P": float(p_val),
                                                                             "val_R": float(r_val),
                                                                             "val_F1": float(f1_val)},
                "test": {"P": float(p_test), "R": float(r_test), "F1": float(f1_test), "TP": int(tp), "FP": int(fp),
                         "FN": int(fn), "TN": int(tn)}}

            exit()



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
                raise ValueError("df_train is empty -> cannot train.")
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

            # ============================================================
            # 0) ADD CLASS WEIGHTS (helps recall)
            # ============================================================
            # Weight anomalies higher; auto-scale by imbalance with a cap
            n_pos = df_train.filter(col(LABEL_COL) == 1).count()
            n_neg = df_train.filter(col(LABEL_COL) == 0).count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Training data must contain both classes (0 and 1) for supervised training.")

            pos_w = min(3.0, float(n_neg) / float(n_pos))  # cap at 10
            print(f"[INFO] Class weight for anomalies (label=1): {pos_w:.3f}")

            df_train_w = df_train.withColumn("class_weight",
                when(col(LABEL_COL) == 1, lit(pos_w)).otherwise(lit(1.0))).cache()

            # ============================================================
            # 1) Fit/tune base models
            # ============================================================

            # ---- Logistic Regression (weighted) ----
            lr = SparkLogisticRegression(featuresCol=FEAT_COL, labelCol=LABEL_COL, maxIter=50, weightCol="class_weight")
            lr_grid = (ParamGridBuilder().addGrid(lr.regParam, [1e-4, 1e-3, 1e-2]).addGrid(lr.elasticNetParam,
                                                                                           [0.0, 0.5, 1.0]).build())
            lr_model = tune_model(lr, lr_grid, df_train_w, evaluator_pr)

            # ---- Random Forest ----
            rf = SparkRandomForestClassifier(featuresCol=FEAT_COL, labelCol=LABEL_COL, seed=42)
            rf_grid = (ParamGridBuilder().addGrid(rf.numTrees, [100, 200]).addGrid(rf.maxDepth, [5, 10]).build())
            rf_model = tune_model(rf, rf_grid, df_train, evaluator_pr)

            # ---- Gradient Boosted Trees (slightly stronger but still fast) ----
            gbt = SparkGBTClassifier(featuresCol=FEAT_COL, labelCol=LABEL_COL, seed=42)
            gbt_grid = (ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5])  # added 7
                        .addGrid(gbt.maxIter, [30, 60])  # keep short runtime
                        .build())
            gbt_model = tune_model(gbt, gbt_grid, df_train, evaluator_pr)

            print("[OK] Base models tuned and trained.")

            # ============================================================
            # 2) Validate: weights by AUC-PR (include recall-friendly sets)
            # ============================================================
            def score_with_p1(model, df, p_col_name):
                return model.transform(df).select(ID_COL, LABEL_COL,
                    vector_to_array("probability")[1].alias(p_col_name))

            val_lr = score_with_p1(lr_model, df_val, "p_lr")
            val_rf = score_with_p1(rf_model, df_val, "p_rf").select(ID_COL, "p_rf")
            val_gbt = score_with_p1(gbt_model, df_val, "p_gbt").select(ID_COL, "p_gbt")

            val_scored = val_lr.join(val_rf, ID_COL, "inner").join(val_gbt, ID_COL, "inner").cache()

            if val_scored.rdd.isEmpty():
                raise ValueError("val_scored is empty after joins. Check Node_block_id consistency/duplicates.")

            @udf(VectorUDT())
            def to_raw_vec(p):
                p = float(p)
                return Vectors.dense([1.0 - p, p])

            # Added GBT-heavy (recall-friendly) options
            weight_sets = [(0.34, 0.33, 0.33), (0.20, 0.20, 0.60), (0.20, 0.60, 0.20), (0.60, 0.20, 0.20),
                (1 / 3, 1 / 3, 1 / 3), ]

            best_w, best_aucpr = None, -1.0
            for w_lr, w_rf, w_gbt in weight_sets:
                tmp = (val_scored.withColumn("p_ens",
                                             w_lr * col("p_lr") + w_rf * col("p_rf") + w_gbt * col("p_gbt")).withColumn(
                    "rawPrediction", to_raw_vec(col("p_ens"))))
                if tmp.rdd.isEmpty():
                    continue
                aucpr = evaluator_pr.evaluate(tmp)
                if aucpr > best_aucpr:
                    best_aucpr = aucpr
                    best_w = (w_lr, w_rf, w_gbt)

            if best_w is None:
                raise ValueError("Could not compute AUC-PR on validation. Ensure VAL has both classes 0 and 1.")

            W_LR, W_RF, W_GBT = best_w
            print(f"[VAL] Best weights (LR, RF, GBT) = {best_w} | AUC-PR = {best_aucpr:.6f}")

            val_ens = val_scored.withColumn("p_ens",
                W_LR * col("p_lr") + W_RF * col("p_rf") + W_GBT * col("p_gbt")).cache()

            # ============================================================
            # 3) Validate: choose threshold to IMPROVE RECALL (subject to precision)
            # ============================================================
            def prf_at_threshold(df_with_p, thr):
                pred = df_with_p.select(col(LABEL_COL).cast("double").alias("label"),
                    when(col("p_ens") >= lit(thr), 1.0).otherwise(0.0).alias("prediction"))
                rdd = pred.rdd.map(lambda r: (r["prediction"], r["label"]))
                m = MulticlassMetrics(rdd)
                return m.precision(1.0), m.recall(1.0), m.fMeasure(1.0)

            P_MIN = 0.85  # keep alerts clean; reduce to 0.93 if you need more recall
            thresholds = [i / 100 for i in range(5, 90)]  # avoid extremes

            best_thr, best_recall, best_f1_at_thr, best_p_at_thr = None, -1.0, -1.0, None
            for t in thresholds:
                p, r, f1v = prf_at_threshold(val_ens, t)
                if p >= P_MIN and r > best_recall:
                    best_recall = r
                    best_thr = t
                    best_f1_at_thr = f1v
                    best_p_at_thr = p

            # Fallback to max-F1 if constraint cannot be satisfied
            if best_thr is None:
                best_thr, best_f1 = None, -1.0
                for t in thresholds:
                    p, r, f1v = prf_at_threshold(val_ens, t)
                    if f1v > best_f1:
                        best_f1 = f1v
                        best_thr = t
                        best_p_at_thr = p
                        best_recall = r
                        best_f1_at_thr = f1v
                print(f"[VAL] Precision constraint not met; using max-F1 threshold.")

            print(
                f"[VAL] Selected threshold = {best_thr:.2f} | Precision={best_p_at_thr:.4f} Recall={best_recall:.4f} F1={best_f1_at_thr:.4f}")

            # ============================================================
            # 4) Test: Precision/Recall/F1
            # ============================================================
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

            print("\nConfusion Matrix (rows=pred, cols=true):")
            print(metrics.confusionMatrix())

        else:

            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # ---- Basic checks ----
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("train", df_final_train_cls), ("val", df_val_cls), ("test", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_final_train_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_val = df_val_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_test = df_test_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            # ---- Class weights ----
            n_pos = df_train.filter(col(LABEL_COL) == 1).count()
            n_neg = df_train.filter(col(LABEL_COL) == 0).count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Train must contain both classes 0 and 1.")

            pos_w = min(10.0, float(n_neg) / float(n_pos))
            df_train_w = df_train.withColumn("class_weight",
                when(col(LABEL_COL) == 1, lit(pos_w)).otherwise(lit(1.0))).cache()

            print(f"[INFO] pos_w={pos_w:.3f}")
            df_train_w.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            # ---- TrainValidationSplit ----
            svm = LinearSVC(featuresCol=FEAT_COL, labelCol=LABEL_COL, weightCol="class_weight", maxIter=120)
            grid = ParamGridBuilder().addGrid(svm.regParam, [1e-5, 1e-4, 1e-3, 1e-2]).build()

            evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            tvs = TrainValidationSplit(estimator=svm, estimatorParamMaps=grid, evaluator=evaluator, trainRatio=0.8,
                parallelism=4)

            # ============================================================
            # Helpers
            # ============================================================
            def prf_at_threshold_fast(df, thr):
                tmp = df.select(col("y").alias("y"), when(col("s1") >= lit(thr), 1).otherwise(0).alias("yhat"))
                agg = tmp.agg(F.sum(((col("yhat") == 1) & (col("y") == 1)).cast("int")).alias("tp"),
                    F.sum(((col("yhat") == 1) & (col("y") == 0)).cast("int")).alias("fp"),
                    F.sum(((col("yhat") == 0) & (col("y") == 1)).cast("int")).alias("fn"),
                    F.sum(((col("yhat") == 0) & (col("y") == 0)).cast("int")).alias("tn"), ).collect()[0]

                tp = int(agg["tp"]);
                fp = int(agg["fp"]);
                fn = int(agg["fn"]);
                tn = int(agg["tn"])
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r / (p + r + 1e-9)
                return p, r, f1, tp, fp, fn, tn

            def score_df(model, df_in):
                return (model.transform(df_in).select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias("y"),
                    vector_to_array(col("rawPrediction"))[1].alias("s1")).cache())

            def best_threshold_fbeta(val_scored, beta=0.5):
                b2 = beta * beta
                qs = [i / 500 for i in range(1, 500)]  # 0.002..0.998
                cand_thr = val_scored.approxQuantile("s1", qs, 0.001)

                best_thr, best_fbeta = None, -1.0
                best_p = best_r = best_f1 = None
                for t in cand_thr:
                    p, r, f1v, *_ = prf_at_threshold_fast(val_scored, float(t))
                    fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-9)
                    if fbeta > best_fbeta:
                        best_fbeta = fbeta
                        best_thr = float(t)
                        best_p, best_r, best_f1 = float(p), float(r), float(f1v)
                return best_thr, best_p, best_r, best_f1, best_fbeta

            # ============================================================
            # 1) Train (Round 0)
            # ============================================================
            svm_model = tvs.fit(df_train_w).bestModel
            print("[OK] Trained LinearSVC (Round 0).")

            # ============================================================
            # 2) SELF-TRAINING to reduce FP ceiling (classification-only)
            # ============================================================
            SELF_ITERS = 2
            POS_Q = 0.995  # stricter positives => higher precision
            NEG_Q = 0.50  # confident negatives

            df_refined = df_train.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            for it in range(SELF_ITERS):
                print(f"\n[SELF-TRAIN] Iteration {it + 1}/{SELF_ITERS} (POS_Q={POS_Q}, NEG_Q={NEG_Q})")

                tr_scored = score_df(svm_model, df_refined)

                thr_pos = float(tr_scored.approxQuantile("s1", [POS_Q], 0.001)[0])
                thr_neg = float(tr_scored.approxQuantile("s1", [NEG_Q], 0.001)[0])

                confident = (tr_scored.withColumn("y_new",
                    when(col("s1") >= lit(thr_pos), lit(1)).when(col("s1") <= lit(thr_neg), lit(0)).otherwise(
                        lit(None))).dropna(subset=["y_new"]).select(ID_COL, FEAT_COL, col("y_new").cast("int").alias(
                    LABEL_COL)).dropDuplicates([ID_COL]).cache())

                print(f"[SELF-TRAIN] thr_pos={thr_pos:.6f} thr_neg={thr_neg:.6f}")
                confident.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

                n_pos_r = confident.filter(col(LABEL_COL) == 1).count()
                n_neg_r = confident.filter(col(LABEL_COL) == 0).count()
                if n_pos_r == 0 or n_neg_r == 0:
                    print("[SELF-TRAIN] Not enough confident samples (one class missing). Stop.")
                    break

                pos_w_r = min(10.0, float(n_neg_r) / float(n_pos_r))
                confident_w = confident.withColumn("class_weight",
                    when(col(LABEL_COL) == 1, lit(pos_w_r)).otherwise(lit(1.0))).cache()

                svm_model = tvs.fit(confident_w).bestModel
                print(f"[OK] Retrained LinearSVC (pos_w={pos_w_r:.3f}).")

            # ============================================================
            # 3) Score VAL
            # ============================================================
            val_scored = score_df(svm_model, df_val).select("y", "s1").cache()

            # ---- Candidate thresholds via quantiles (for constraints check) ----
            qs = [i / 200 for i in range(1, 200)]  # 0.005..0.995
            cand_thr = val_scored.approxQuantile("s1", qs, 0.001)

            # ============================================================
            # 4) Threshold selection (constraints first, fallback = F0.5)
            # ============================================================
            P_MIN = 0.90
            R_MIN = 0.90

            best_thr = None
            best_f1 = -1.0
            best_p = best_r = best_f1_at = None

            for t in cand_thr:
                p, r, f1v, *_ = prf_at_threshold_fast(val_scored, float(t))
                if p >= P_MIN and r >= R_MIN:
                    if f1v > best_f1:
                        best_f1 = f1v
                        best_thr = float(t)
                        best_p, best_r, best_f1_at = float(p), float(r), float(f1v)

            # fallback: best F0.5 (precision-focused)
            if best_thr is None:
                FBETA = 0.5
                best_thr, best_p, best_r, best_f1_at, best_fbeta = best_threshold_fbeta(val_scored, beta=FBETA)
                print(f"[VAL] No threshold satisfied P_MIN={P_MIN} and R_MIN={R_MIN}. Using best F{FBETA} fallback.")

            print(f"\n[VAL] Selected thr={best_thr:.6f} | P={best_p:.4f} R={best_r:.4f} F1={best_f1_at:.4f}")
            print(f"[VAL] Constraints used: P_MIN={P_MIN}, R_MIN={R_MIN}, fallback F0.5")

            # ============================================================
            # 5) Evaluate on TEST
            # ============================================================
            test_scored = score_df(svm_model, df_test).select("y", "s1").cache()
            p_test, r_test, f1_test, tp, fp, fn, tn = prf_at_threshold_fast(test_scored, best_thr)

            print("\n=== TEST METRICS (LinearSVC + Self-Training) ===")
            print(f"Precision (anomaly=1): {p_test:.4f}")
            print(f"Recall    (anomaly=1): {r_test:.4f}")
            print(f"F1-score  (anomaly=1): {f1_test:.4f}")
            print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

            results = {"pos_w": float(pos_w), "thr": float(best_thr),
                "val": {"P": float(best_p), "R": float(best_r), "F1": float(best_f1_at)},
                "test": {"P": float(p_test), "R": float(r_test), "F1": float(f1_test), "TP": int(tp), "FP": int(fp),
                         "FN": int(fn), "TN": int(tn)},
                "self_train": {"iters": SELF_ITERS, "POS_Q": POS_Q, "NEG_Q": NEG_Q}}

            # If you want to stop here
            exit()



            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # ---- Basic checks ----
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("train", df_final_train_cls), ("val", df_val_cls), ("test", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_final_train_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_val = df_val_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_test = df_test_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            # ---- Class weights ----
            n_pos = df_train.filter(col(LABEL_COL) == 1).count()
            n_neg = df_train.filter(col(LABEL_COL) == 0).count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Train must contain both classes 0 and 1.")

            pos_w = min(10.0, float(n_neg) / float(n_pos))  # allow a bit higher; helps hard datasets
            df_train_w = df_train.withColumn("class_weight",
                when(col(LABEL_COL) == 1, lit(pos_w)).otherwise(lit(1.0))).cache()

            print(f"[INFO] pos_w={pos_w:.3f}")
            df_train_w.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            # ---- TrainValidationSplit ----
            svm = LinearSVC(featuresCol=FEAT_COL, labelCol=LABEL_COL, weightCol="class_weight", maxIter=80)
            grid = ParamGridBuilder().addGrid(svm.regParam, [1e-5, 1e-4, 1e-3, 1e-2]).build()

            evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            tvs = TrainValidationSplit(estimator=svm, estimatorParamMaps=grid, evaluator=evaluator, trainRatio=0.8,
                parallelism=4)

            svm_model = tvs.fit(df_train_w).bestModel
            print("[OK] Trained LinearSVC.")

            # ---- Score VAL ----
            val_scored = (svm_model.transform(df_val).select(col(LABEL_COL).cast("int").alias("y"),
                vector_to_array(col("rawPrediction"))[1].alias("s1")).cache())

            # ---- Fast PRF at threshold using aggregations ----
            def prf_at_threshold_fast(df, thr):
                tmp = df.select(col("y").alias("y"), when(col("s1") >= lit(thr), 1).otherwise(0).alias("yhat"))
                agg = tmp.agg(F.sum(((col("yhat") == 1) & (col("y") == 1)).cast("int")).alias("tp"),
                    F.sum(((col("yhat") == 1) & (col("y") == 0)).cast("int")).alias("fp"),
                    F.sum(((col("yhat") == 0) & (col("y") == 1)).cast("int")).alias("fn"), ).collect()[0]

                tp = int(agg["tp"]);
                fp = int(agg["fp"]);
                fn = int(agg["fn"])
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r / (p + r + 1e-9)
                return p, r, f1, tp, fp, fn

            # ---- Candidate thresholds via quantiles ----
            qs = [i / 200 for i in range(1, 200)]  # finer: 0.005..0.995
            cand_thr = val_scored.approxQuantile("s1", qs, 0.001)

            # ============================================================
            # THRESHOLD SELECTION THAT CONSIDERS ALL 3 METRICS
            # Strategy:
            #   1) Find thresholds satisfying both:
            #          Precision >= P_MIN  and  Recall >= R_MIN
            #      pick the one with maximum F1 (or F-beta)
            #   2) If none exist, fallback to max F-beta
            # ============================================================

            # ---- choose targets (dataset-robust defaults) ----
            P_MIN = 0.90  # increase precision on hard sets (SOMA)
            R_MIN = 0.90  # keep recall high enough
            BETA = 1.0  # use F1 inside feasible region; fallback uses F-beta with this beta

            b2 = BETA * BETA

            best_thr = None
            best_f = -1.0
            best_p = best_r = best_f1 = None

            # Keep top candidates for inspection
            topk = []  # list of (score, thr, p, r, f1)

            for t in cand_thr:
                p, r, f1v, tp, fp, fn = prf_at_threshold_fast(val_scored, t)

                # score: F-beta
                fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-9)

                # store for debug
                topk.append((fbeta, float(t), float(p), float(r), float(f1v)))

                # feasible region: meet both constraints
                if p >= P_MIN and r >= R_MIN:
                    # within feasible, maximize F1 (or fbeta)
                    if f1v > best_f:
                        best_f = f1v
                        best_thr, best_p, best_r, best_f1 = float(t), float(p), float(r), float(f1v)

            # fallback if constraints impossible
            if best_thr is None:
                # fallback to max F-beta (same beta as above)
                best_score, best_thr, best_p, best_r, best_f1 = max(topk, key=lambda x: x[0])
                print(f"[VAL] No threshold satisfied P_MIN={P_MIN} and R_MIN={R_MIN}. Using max F{BETA:.1f} fallback.")

            # show top 10 candidate thresholds (optional but useful)
            topk_sorted = sorted(topk, key=lambda x: x[0], reverse=True)[:10]
            print("\n[VAL] Top thresholds by F-beta (showing 10):")
            print("rank | thr      | P      | R      | F1")
            for i, (_, thr, p, r, f1v) in enumerate(topk_sorted, 1):
                print(f"{i:>4} | {thr:>7.4f} | {p:>6.3f} | {r:>6.3f} | {f1v:>6.3f}")

            print(f"\n[VAL] Selected thr={best_thr:.6f} | P={best_p:.4f} R={best_r:.4f} F1={best_f1:.4f}")
            print(f"[VAL] Constraints used: P_MIN={P_MIN}, R_MIN={R_MIN}, fallback F{BETA:.1f}")

            # ---- Evaluate on TEST ----
            test_scored = (svm_model.transform(df_test).select(col(LABEL_COL).cast("int").alias("y"),
                vector_to_array(col("rawPrediction"))[1].alias("s1")).cache())

            p_test, r_test, f1_test, tp, fp, fn = prf_at_threshold_fast(test_scored, best_thr)

            print("\n=== TEST METRICS (LinearSVC single classifier) ===")
            print(f"Precision (anomaly=1): {p_test:.4f}")
            print(f"Recall    (anomaly=1): {r_test:.4f}")
            print(f"F1-score  (anomaly=1): {f1_test:.4f}")
            print(f"TP={tp} FP={fp} FN={fn}")

            # return whatever you need
            results = {"pos_w": float(pos_w), "thr": float(best_thr),
                "val": {"P": float(best_p), "R": float(best_r), "F1": float(best_f1)},
                "test": {"P": float(p_test), "R": float(r_test), "F1": float(f1_test), "TP": int(tp), "FP": int(fp),
                         "FN": int(fn)}}

            exit()

            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # ---- Basic checks ----
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("train", df_final_train_cls), ("val", df_val_cls), ("test", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_final_train_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_val = df_val_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_test = df_test_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            # ---- Class weights ----
            n_pos = df_train.filter(col(LABEL_COL) == 1).count()
            n_neg = df_train.filter(col(LABEL_COL) == 0).count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Train must contain both classes 0 and 1.")

            pos_w = min(5.0, float(n_neg) / float(n_pos))
            df_train_w = df_train.withColumn("class_weight",
                when(col(LABEL_COL) == 1, lit(pos_w)).otherwise(lit(1.0))).cache()

            print(f"[INFO] pos_w={pos_w:.3f}")
            df_train_w.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            # ---- TrainValidationSplit ----
            svm = LinearSVC(featuresCol=FEAT_COL, labelCol=LABEL_COL, weightCol="class_weight", maxIter=50)
            grid = ParamGridBuilder().addGrid(svm.regParam, [1e-4, 1e-3, 1e-2]).build()

            evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            tvs = TrainValidationSplit(estimator=svm, estimatorParamMaps=grid, evaluator=evaluator, trainRatio=0.8,
                parallelism=4)

            svm_model = tvs.fit(df_train_w).bestModel
            print("[OK] Trained LinearSVC.")

            # ---- Score VAL ----
            val_scored = (svm_model.transform(df_val).select(col(LABEL_COL).cast("int").alias("y"),
                                                             vector_to_array(col("rawPrediction"))[1].alias(
                                                                 "s1")).cache())

            # ---- Fast PRF at threshold using aggregations (NO MulticlassMetrics loop) ----
            def prf_at_threshold_fast(df, thr):
                tmp = df.select(col("y").alias("y"), when(col("s1") >= lit(thr), 1).otherwise(0).alias("yhat"))
                agg = tmp.agg(F.sum(((col("yhat") == 1) & (col("y") == 1)).cast("int")).alias("tp"),
                    F.sum(((col("yhat") == 1) & (col("y") == 0)).cast("int")).alias("fp"),
                    F.sum(((col("yhat") == 0) & (col("y") == 1)).cast("int")).alias("fn"), ).collect()[0]

                tp = int(agg["tp"]);
                fp = int(agg["fp"]);
                fn = int(agg["fn"])
                p = tp / (tp + fp + 1e-9)
                r = tp / (tp + fn + 1e-9)
                f1 = 2 * p * r / (p + r + 1e-9)
                return p, r, f1

            # ---- Candidate thresholds via quantiles ----
            qs = [i / 100 for i in range(1, 100)]  # 0.01..0.99
            cand_thr = val_scored.approxQuantile("s1", qs, 0.001)

            # ============================================================
            # THRESHOLD SELECTION (IMPROVED)
            # 1) Maximize recall subject to Precision >= P_MIN
            # 2) fallback to F2 (recall-heavy) if constraint not met
            # ============================================================
            P_MIN = 0.95  # keep your improved precision target
            BETA = 2.0  # fallback recall-heavy objective
            b2 = BETA * BETA

            best_thr = None
            best_r = -1.0
            best_p = None
            best_f1 = None

            # 1) constraint mode
            for t in cand_thr:
                p, r, f1v = prf_at_threshold_fast(val_scored, t)
                if p >= P_MIN and r > best_r:
                    best_thr, best_r, best_p, best_f1 = t, r, p, f1v

            # 2) fallback to best F2 if no threshold satisfies precision constraint
            if best_thr is None:
                best_score = -1.0
                for t in cand_thr:
                    p, r, f1v = prf_at_threshold_fast(val_scored, t)
                    fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-9)
                    if fbeta > best_score:
                        best_score = fbeta
                        best_thr, best_r, best_p, best_f1 = t, r, p, f1v
                print("[VAL] No threshold met precision constraint; using best F2 fallback.")

            print(f"[VAL] Selected thr={best_thr:.6f} | P={best_p:.4f} R={best_r:.4f} F1={best_f1:.4f}")

            # ---- Evaluate on TEST ----
            test_scored = (svm_model.transform(df_test).select(col(LABEL_COL).cast("int").alias("y"),
                                                               vector_to_array(col("rawPrediction"))[1].alias(
                                                                   "s1")).cache())

            p_test, r_test, f1_test = prf_at_threshold_fast(test_scored, best_thr)

            print("\n=== TEST METRICS (LinearSVC single classifier) ===")
            print(f"Precision (anomaly=1): {p_test:.4f}")
            print(f"Recall    (anomaly=1): {r_test:.4f}")
            print(f"F1-score  (anomaly=1): {f1_test:.4f}")


            exit()
            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # ---- Basic checks ----
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("train", df_final_train_cls), ("val", df_val_cls), ("test", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_final_train_cls.select(ID_COL, FEAT_COL, LABEL_COL).cache()
            df_val = df_val_cls.select(ID_COL, FEAT_COL, LABEL_COL).cache()
            df_test = df_test_cls.select(ID_COL, FEAT_COL, LABEL_COL).cache()

            # ---- Class weights (cheap & effective) ----
            n_pos = df_train.filter(col(LABEL_COL) == 1).count()
            n_neg = df_train.filter(col(LABEL_COL) == 0).count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Train must contain both classes 0 and 1.")

            pos_w = min(5.0, float(n_neg) / float(n_pos))  # cap to keep precision stable
            df_train_w = df_train.withColumn("class_weight",
                when(col(LABEL_COL) == 1, lit(pos_w)).otherwise(lit(1.0))).cache()

            print(f"[INFO] pos_w={pos_w:.3f}")
            df_train_w.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            # ---- TrainValidationSplit (fast) ----
            # LinearSVC outputs `rawPrediction` but not probability. We'll threshold raw score.
            svm = LinearSVC(featuresCol=FEAT_COL, labelCol=LABEL_COL, weightCol="class_weight", maxIter=50)

            # small grid -> short runtime
            grid = (ParamGridBuilder().addGrid(svm.regParam, [1e-4, 1e-3, 1e-2]).build())

            # Use areaUnderROC on rawPrediction for tuning (works well for score-based models)
            evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            tvs = TrainValidationSplit(estimator=svm, estimatorParamMaps=grid, evaluator=evaluator, trainRatio=0.8,
                parallelism=4)
            svm_model = tvs.fit(df_train_w).bestModel
            print("[OK] Trained LinearSVC.")

            # ---- Score VAL ----
            val_scored = svm_model.transform(df_val).select(col(LABEL_COL).cast("double").alias("label"),
                col("rawPrediction"))

            # rawPrediction is a vector of length 2, take score for class 1
            from pyspark.ml.functions import vector_to_array
            val_scored = val_scored.withColumn("s1", vector_to_array("rawPrediction")[1]).cache()

            def prf_at_threshold(df, thr):
                pred = df.select(col("label"), when(col("s1") >= lit(thr), 1.0).otherwise(0.0).alias("prediction"))
                rdd = pred.rdd.map(lambda r: (r["prediction"], r["label"]))
                m = MulticlassMetrics(rdd)
                return m.precision(1.0), m.recall(1.0), m.fMeasure(1.0)

            # ---- Choose threshold on VAL by best F1 ----
            # Use score quantiles to search quickly (fast & stable)
            qs = [i / 100 for i in range(5, 96, 5)]  # 0.05..0.95
            cand_thr = val_scored.approxQuantile("s1", qs, 0.001)

            best_thr, best_f1, best_p, best_r = None, -1.0, None, None
            for t in cand_thr:
                p, r, f1v = prf_at_threshold(val_scored, t)
                if f1v > best_f1:
                    best_f1, best_thr, best_p, best_r = f1v, t, p, r

            print(f"[VAL] Best thr={best_thr:.6f} | P={best_p:.4f} R={best_r:.4f} F1={best_f1:.4f}")

            # ---- Evaluate on TEST ----
            test_scored = svm_model.transform(df_test).select(col(LABEL_COL).cast("double").alias("label"),
                vector_to_array("rawPrediction")[1].alias("s1"))

            pred_test = test_scored.select(col("label"),
                when(col("s1") >= lit(best_thr), 1.0).otherwise(0.0).alias("prediction"))

            rdd_test = pred_test.rdd.map(lambda r: (r["prediction"], r["label"]))
            m = MulticlassMetrics(rdd_test)

            print("\n=== TEST METRICS (LinearSVC single classifier) ===")
            print(f"Precision (anomaly=1): {m.precision(1.0):.4f}")
            print(f"Recall    (anomaly=1): {m.recall(1.0):.4f}")
            print(f"F1-score  (anomaly=1): {m.fMeasure(1.0):.4f}")
