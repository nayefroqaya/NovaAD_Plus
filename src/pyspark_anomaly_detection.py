
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



        else:

            from pyspark.ml.feature import VectorAssembler
            from pyspark.sql import functions as F
            from pyspark.sql.functions import col, lit, when
            from pyspark.ml.classification import LinearSVC
            from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
            from pyspark.ml.evaluation import BinaryClassificationEvaluator
            from pyspark.ml.functions import vector_to_array
            import time

            # -----------------------------
            # CONFIG: column names
            # -----------------------------
            LABEL_COL = "Final_Label"
            ID_COL = "Node_block_id"

            # ---- NEW: feature columns used by classifier ----
            PCA_VEC_COL = "pca_features"  # vector
            SCORE_PCA_COL = "anomaly_score_pca"  # scalar
            SCORE_GMM_COL = "anomaly_score_gmm"  # scalar
            COMB_COL = "comb_score"  # scalar
            CLS_FEAT_COL = "cls_features"  # vector output for classifier

            # -----------------------------
            # Helper: Fast PRF at threshold
            # -----------------------------
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

            # -------------------------------------------------------
            # Threshold selection on VAL (constraints + fallback)
            # -------------------------------------------------------
            def select_threshold_from_val(val_predictions_df, label_col=LABEL_COL, raw_pred_col="rawPrediction",
                    pos_index=1, P_MIN=0.90, R_MIN=0.90, BETA=1.0, quantile_step=0.005, quantile_rel_err=0.001):
                # rawPrediction for LinearSVC is a vector; we take pos_index component
                val_scored = (val_predictions_df.select(col(label_col).cast("int").alias("y"),
                    vector_to_array(col(raw_pred_col))[pos_index].alias("s1")).cache())
                _ = val_scored.count()

                qs = [i * quantile_step for i in range(1, int(1 / quantile_step))]
                cand_thr = val_scored.approxQuantile("s1", qs, quantile_rel_err)

                b2 = BETA * BETA
                best_thr = None
                best_f = -1.0
                best_p = best_r = best_f1 = None
                topk = []  # (fbeta, thr, p, r, f1)

                for t in cand_thr:
                    p, r, f1v, tp, fp, fn = prf_at_threshold_fast(val_scored, thr=t, label_col="y", score_col="s1")
                    fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-9)
                    topk.append((float(fbeta), float(t), float(p), float(r), float(f1v)))

                    if p >= P_MIN and r >= R_MIN:
                        if f1v > best_f:
                            best_f = f1v
                            best_thr, best_p, best_r, best_f1 = float(t), float(p), float(r), float(f1v)

                fallback_used = False
                if best_thr is None:
                    best_score, best_thr, best_p, best_r, best_f1 = max(topk, key=lambda x: x[0])
                    fallback_used = True

                topk_sorted = sorted(topk, key=lambda x: x[0], reverse=True)[:10]

                return {"thr": float(best_thr), "val": {"P": float(best_p), "R": float(best_r), "F1": float(best_f1)},
                    "fallback_used": fallback_used, "top10": topk_sorted}

            # ============================================================
            # MAIN: assumes these DataFrames already exist:
            #   df_final_train_cls, df_val_cls, df_test_cls
            # AND they SHOULD contain anomaly_score_pca/anomaly_score_gmm
            # ============================================================

            # ---- Ensure comb_score exists everywhere ----
            def ensure_scores(df):
                out = df
                if COMB_COL not in out.columns:
                    # require both scores
                    out = out.withColumn(COMB_COL, col(SCORE_PCA_COL) + col(SCORE_GMM_COL))
                return out

            df_train0 = ensure_scores(df_final_train_cls)
            df_val0 = ensure_scores(df_val_cls)
            df_test0 = ensure_scores(df_test_cls)

            # ---- Required columns check (UPDATED) ----
            required = {ID_COL, PCA_VEC_COL, SCORE_PCA_COL, SCORE_GMM_COL, COMB_COL, LABEL_COL}
            for name, df in [("train", df_train0), ("val", df_val0), ("test", df_test0)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            # ---- Assemble classifier features: [pca_features + 3 score scalars] ----
            assembler = VectorAssembler(inputCols=[PCA_VEC_COL, SCORE_PCA_COL, SCORE_GMM_COL, COMB_COL],
                outputCol=CLS_FEAT_COL)

            df_train = (assembler.transform(df_train0).select(ID_COL, col(LABEL_COL).cast("int").alias(LABEL_COL),
                                                              col(CLS_FEAT_COL).alias(CLS_FEAT_COL)).cache())
            df_val = (assembler.transform(df_val0).select(ID_COL, col(LABEL_COL).cast("int").alias(LABEL_COL),
                                                          col(CLS_FEAT_COL).alias(CLS_FEAT_COL)).cache())
            df_test = (assembler.transform(df_test0).select(ID_COL, col(LABEL_COL).cast("int").alias(LABEL_COL),
                                                            col(CLS_FEAT_COL).alias(CLS_FEAT_COL)).cache())

            _ = df_train.count()
            _ = df_val.count()
            _ = df_test.count()

            # ---- Class weights ----
            n_pos = df_train.filter(col(LABEL_COL) == 1).count()
            n_neg = df_train.filter(col(LABEL_COL) == 0).count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Train must contain both classes 0 and 1.")

            pos_w = min(10.0, float(n_neg) / float(n_pos))
            df_train_w = (
                df_train.withColumn("class_weight", when(col(LABEL_COL) == 1, lit(pos_w)).otherwise(lit(1.0))).cache())

            print(f"[INFO] pos_w={pos_w:.3f}")
            df_train_w.groupBy(LABEL_COL).count().orderBy(LABEL_COL).show()

            # ---- TrainValidationSplit ----
            svm = LinearSVC(featuresCol=CLS_FEAT_COL,  # UPDATED
                labelCol=LABEL_COL, weightCol="class_weight", maxIter=60)

            grid = (ParamGridBuilder().addGrid(svm.regParam, [1e-5, 1e-4, 1e-3]).build())

            evaluator_auc = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            tvs = TrainValidationSplit(estimator=svm, estimatorParamMaps=grid, evaluator=evaluator_auc, trainRatio=0.8,
                parallelism=4)

            # -----------------------------
            # TRAIN runtime (fit)
            # -----------------------------
            t0 = time.perf_counter()
            tvs_model = tvs.fit(df_train_w)
            train_runtime_sec = time.perf_counter() - t0
            train_runtime_min = train_runtime_sec / 60.0

            svm_model = tvs_model.bestModel
            print("[OK] Trained LinearSVC (with scores in features).")
            print(f"[TIME] train_runtime_min={train_runtime_min:.3f}")

            # -----------------------------
            # VAL threshold selection
            # -----------------------------
            val_pred = svm_model.transform(df_val).cache()
            _ = val_pred.count()

            thr_info = select_threshold_from_val(val_pred, label_col=LABEL_COL, raw_pred_col="rawPrediction",
                pos_index=1, P_MIN=0.90, R_MIN=0.90, BETA=1.0, quantile_step=0.005, quantile_rel_err=0.001)

            best_thr = thr_info["thr"]
            best_p = thr_info["val"]["P"]
            best_r = thr_info["val"]["R"]
            best_f1 = thr_info["val"]["F1"]

            if thr_info["fallback_used"]:
                print("[VAL] No threshold satisfied constraints; using max F-beta fallback.")
            print("\n[VAL] Top thresholds by F-beta (top 10):")
            print("rank | thr      | P      | R      | F1")
            for i, (fbeta, thr, p, r, f1v) in enumerate(thr_info["top10"], 1):
                print(f"{i:>4} | {thr:>7.4f} | {p:>6.3f} | {r:>6.3f} | {f1v:>6.3f}")

            print(f"\n[VAL] Selected thr={best_thr:.6f} | P={best_p:.4f} R={best_r:.4f} F1={best_f1:.4f}")

            # -----------------------------
            # TEST runtime (transform)
            # -----------------------------
            t1 = time.perf_counter()
            test_pred = svm_model.transform(df_test).cache()
            _ = test_pred.count()
            test_runtime_sec = time.perf_counter() - t1
            test_runtime_min = test_runtime_sec / 60.0

            return test_pred, LABEL_COL, best_thr, train_runtime_min, test_runtime_min









            # -----------------------------1
            # CONFIG: column names
            # -----------------------------
            LABEL_COL = "Final_Label"
            FEAT_COL = "pca_features"
            ID_COL = "Node_block_id"

            # -----------------------------
            # Helper: Fast PRF at threshold
            # -----------------------------
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

            # -------------------------------------------------------
            # Threshold selection on VAL (constraints + fallback)
            # -------------------------------------------------------
            def select_threshold_from_val(val_predictions_df, label_col=LABEL_COL, raw_pred_col="rawPrediction",
                    pos_index=1, P_MIN=0.90, R_MIN=0.90, BETA=1.0, quantile_step=0.005,  # 0.005 => 199 candidates
                    quantile_rel_err=0.001):
                val_scored = (val_predictions_df.select(col(label_col).cast("int").alias("y"),
                    vector_to_array(col(raw_pred_col))[pos_index].alias("s1")).cache())
                _ = val_scored.count()

                qs = [i * quantile_step for i in range(1, int(1 / quantile_step))]
                cand_thr = val_scored.approxQuantile("s1", qs, quantile_rel_err)

                b2 = BETA * BETA
                best_thr = None
                best_f = -1.0
                best_p = best_r = best_f1 = None

                topk = []  # (fbeta, thr, p, r, f1)

                for t in cand_thr:
                    p, r, f1v, tp, fp, fn = prf_at_threshold_fast(val_scored, thr=t, label_col="y", score_col="s1")
                    fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-9)

                    topk.append((float(fbeta), float(t), float(p), float(r), float(f1v)))

                    if p >= P_MIN and r >= R_MIN:
                        # within feasible region, maximize F1
                        if f1v > best_f:
                            best_f = f1v
                            best_thr, best_p, best_r, best_f1 = float(t), float(p), float(r), float(f1v)

                # fallback if no feasible threshold exists
                fallback_used = False
                if best_thr is None:
                    best_score, best_thr, best_p, best_r, best_f1 = max(topk, key=lambda x: x[0])
                    fallback_used = True

                # top 10 by F-beta for debugging
                topk_sorted = sorted(topk, key=lambda x: x[0], reverse=True)[:10]

                return {"thr": float(best_thr), "val": {"P": float(best_p), "R": float(best_r), "F1": float(best_f1)},
                    "fallback_used": fallback_used, "top10": topk_sorted}

            # ============================================================
            # MAIN: assumes these DataFrames already exist:
            #   df_final_train_cls, df_val_cls, df_test_cls
            # ============================================================

            # ---- Basic checks ----
            required = {ID_COL, FEAT_COL, LABEL_COL}
            for name, df in [("train", df_final_train_cls), ("val", df_val_cls), ("test", df_test_cls)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            df_train = df_final_train_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_val = df_val_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()
            df_test = df_test_cls.select(ID_COL, FEAT_COL, col(LABEL_COL).cast("int").alias(LABEL_COL)).cache()

            # materialize caches (optional but makes timings more stable)
            _ = df_train.count()
            _ = df_val.count()
            _ = df_test.count()

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
            svm = LinearSVC(featuresCol=FEAT_COL, labelCol=LABEL_COL, weightCol="class_weight", maxIter=60)  # 80

            grid = ParamGridBuilder().addGrid(svm.regParam, [1e-5, 1e-4, 1e-3]).build()   # 1e-5, 1e-4, 1e-3, 1e-2

            evaluator_auc = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            tvs = TrainValidationSplit(estimator=svm, estimatorParamMaps=grid, evaluator=evaluator_auc, trainRatio=0.8,
                parallelism=4)

            # -----------------------------
            # TRAIN runtime (fit)
            # -----------------------------
            t0 = time.perf_counter()
            tvs_model = tvs.fit(df_train_w)
            train_runtime_sec = time.perf_counter() - t0
            train_runtime_sec = train_runtime_sec
            train_runtime_min = train_runtime_sec / 60.0

            svm_model = tvs_model.bestModel
            print("[OK] Trained LinearSVC.")
            print(f"[TIME] train_runtime_sec={train_runtime_min:.3f}")

            # -----------------------------
            # VAL threshold selection
            # -----------------------------
            val_pred = svm_model.transform(df_val).cache()
            _ = val_pred.count()

            thr_info = select_threshold_from_val(val_pred, label_col=LABEL_COL, raw_pred_col="rawPrediction",
                pos_index=1, P_MIN=0.90, R_MIN=0.90, BETA=1.0, quantile_step=0.005, quantile_rel_err=0.001)

            best_thr = thr_info["thr"]
            best_p = thr_info["val"]["P"]
            best_r = thr_info["val"]["R"]
            best_f1 = thr_info["val"]["F1"]

            if thr_info["fallback_used"]:
                print(f"[VAL] No threshold satisfied constraints; using max F-beta fallback.")
            print("\n[VAL] Top thresholds by F-beta (top 10):")
            print("rank | thr      | P      | R      | F1")
            for i, (fbeta, thr, p, r, f1v) in enumerate(thr_info["top10"], 1):
                print(f"{i:>4} | {thr:>7.4f} | {p:>6.3f} | {r:>6.3f} | {f1v:>6.3f}")

            print(f"\n[VAL] Selected thr={best_thr:.6f} | P={best_p:.4f} R={best_r:.4f} F1={best_f1:.4f}")

            # -----------------------------
            # TEST runtime (transform + eval)
            # -----------------------------
            t1 = time.perf_counter()
            test_pred = svm_model.transform(df_test).cache()
            _ = test_pred.count()  # materialize transform
            test_runtime_sec = time.perf_counter() - t1
            test_runtime_sec = test_runtime_sec
            test_runtime_min = test_runtime_sec / 60.0


            return  test_pred, LABEL_COL , best_thr ,  train_runtime_min, test_runtime_min

            test_metrics = evaluation_pyspark(test_pred, label_col=LABEL_COL, raw_pred_col="rawPrediction",
                                              thr=best_thr, pos_index=1)
            test_runtime_sec = time.perf_counter() - t1
            test_runtime_sec = test_runtime_sec
            test_runtime_min = test_runtime_sec / 60.0

            print("\n=== TEST METRICS (LinearSVC single classifier) ===")
            print(f"Precision (anomaly=1): {test_metrics['P']:.4f}")
            print(f"Recall    (anomaly=1): {test_metrics['R']:.4f}")
            print(f"F1-score  (anomaly=1): {test_metrics['F1']:.4f}")
            print(f"TP={test_metrics['TP']} FP={test_metrics['FP']} FN={test_metrics['FN']}")
            print(f"[TIME] test_runtime_sec={test_runtime_min:.3f}")

            # -----------------------------
            # Final results dict (includes runtimes)
            # -----------------------------
            results = {"pos_w": float(pos_w), "thr": float(best_thr), "train_runtime_sec": float(train_runtime_sec),
                "train_runtime_min": float(train_runtime_min), "test_runtime_sec": float(test_runtime_sec),
                "test_runtime_min": float(test_runtime_min),
                "val": {"P": float(best_p), "R": float(best_r), "F1": float(best_f1)},
                "test": {"P": float(test_metrics["P"]), "R": float(test_metrics["R"]), "F1": float(test_metrics["F1"]),
                    "TP": int(test_metrics["TP"]), "FP": int(test_metrics["FP"]), "FN": int(test_metrics["FN"]), }, }


            print("\n[RESULTS]")
            print(results)
            print(f"[TIME] test_runtime_sec={test_runtime_min:.3f}")
            print(f"[TIME] train_runtime_sec={train_runtime_min:.3f}")


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
