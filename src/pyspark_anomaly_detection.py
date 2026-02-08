
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

            # -----------------------------
            # Column names
            # -----------------------------
            ID_COL = "Node_block_id"
            FEAT_COL = "pca_features"
            TRAIN_LABEL_COL = "Final_Label"  # pseudo labels for training
            EVAL_LABEL_TMP = "y_eval"  # internal eval label name

            # -----------------------------
            # Checks
            # -----------------------------
            for name, df, req in [("df_final_train_cls", df_final_train_cls, {ID_COL, FEAT_COL, TRAIN_LABEL_COL}),
                ("df_val_cls", df_val_cls, {ID_COL, FEAT_COL}), ("df_test_cls", df_test_cls, {ID_COL, FEAT_COL}), ]:
                missing = req - set(df.columns)
                if missing:
                    raise ValueError(f"{name} missing columns: {missing}")

            # ============================================================
            # 1) TRAIN set (pseudo labels only)
            # ============================================================
            df_train = (df_final_train_cls.select(ID_COL, FEAT_COL,
                                                  col(TRAIN_LABEL_COL).cast("int").alias(TRAIN_LABEL_COL)).cache())
            if df_train.rdd.isEmpty():
                raise ValueError("df_train is empty -> cannot train.")

            print("Train label distribution (pseudo Final_Label):")
            df_train.groupBy(TRAIN_LABEL_COL).count().orderBy(TRAIN_LABEL_COL).show()

            # ============================================================
            # 2) Build EVAL labels for VAL/TEST (map string Label -> 0/1 if exists)
            # ============================================================
            def map_string_label_to_int(df_in):
                # 'normal'/'Normal' -> 0, everything else -> 1
                return (df_in.withColumn("label_str", F.lower(F.trim(col("Label")))).withColumn(EVAL_LABEL_TMP, when(
                    col("label_str").isin("normal", "0", "false"), lit(0)).otherwise(lit(1)).cast("int")))

            def build_eval_df(df_in, name):
                cols = set(df_in.columns)
                if "Label" in cols:
                    df_out = (map_string_label_to_int(df_in).select(ID_COL, FEAT_COL,
                                                                    col(EVAL_LABEL_TMP).alias(TRAIN_LABEL_COL)).cache())
                    print(f"[INFO] {name}: using ground-truth string Label mapped to 0/1.")
                    return df_out
                if TRAIN_LABEL_COL in cols:
                    df_out = (
                        df_in.select(ID_COL, FEAT_COL, col(TRAIN_LABEL_COL).cast("int").alias(TRAIN_LABEL_COL)).cache())
                    print(f"[INFO] {name}: no Label column; using Final_Label for evaluation.")
                    return df_out
                raise ValueError(f"{name} has neither 'Label' nor '{TRAIN_LABEL_COL}' to evaluate.")

            df_val = build_eval_df(df_val_cls, "VAL")
            df_test = build_eval_df(df_test_cls, "TEST")

            if df_val.rdd.isEmpty():
                raise ValueError("df_val is empty -> cannot validate.")
            if df_test.rdd.isEmpty():
                raise ValueError("df_test is empty -> cannot test.")

            print("VAL label distribution (evaluation labels):")
            df_val.groupBy(TRAIN_LABEL_COL).count().orderBy(TRAIN_LABEL_COL).show()

            print("TEST label distribution (evaluation labels):")
            df_test.groupBy(TRAIN_LABEL_COL).count().orderBy(TRAIN_LABEL_COL).show()

            # ============================================================
            # 3) Clean training set (classification-only)
            #    deterministic negative downsample, keep all positives
            # ============================================================
            df_pos = df_train.filter(col(TRAIN_LABEL_COL) == 1)
            df_neg = df_train.filter(col(TRAIN_LABEL_COL) == 0)

            n_pos = df_pos.count()
            n_neg = df_neg.count()
            if n_pos == 0 or n_neg == 0:
                raise ValueError("Training must contain both classes 0 and 1.")

            NEG_RATIO = 8
            max_neg = int(NEG_RATIO * n_pos)

            if n_neg > max_neg:
                df_neg = (df_neg.withColumn("hid", F.xxhash64(col(ID_COL))).orderBy("hid").limit(max_neg).drop("hid"))
                print(f"[INFO] Downsampled negatives: {n_neg} -> {max_neg}")

            df_train_clean = df_neg.unionByName(df_pos).cache()

            print("Train label distribution (clean):")
            df_train_clean.groupBy(TRAIN_LABEL_COL).count().orderBy(TRAIN_LABEL_COL).show()

            # ============================================================
            # 4) Class weights (cap 20)
            # ============================================================
            n_pos_c = df_train_clean.filter(col(TRAIN_LABEL_COL) == 1).count()
            n_neg_c = df_train_clean.filter(col(TRAIN_LABEL_COL) == 0).count()
            pos_w = min(20.0, float(n_neg_c) / float(n_pos_c))
            print(f"[INFO] Class weight for anomalies (label=1): {pos_w:.3f}")

            df_train_w = (df_train_clean.withColumn("class_weight",
                                                    when(col(TRAIN_LABEL_COL) == 1, lit(pos_w)).otherwise(
                                                        lit(1.0))).cache())

            # ============================================================
            # 5) Evaluator and tuning helper
            # ============================================================
            evaluator_pr = BinaryClassificationEvaluator(labelCol=TRAIN_LABEL_COL, rawPredictionCol="rawPrediction",
                metricName="areaUnderPR")

            def tune_model(estimator, param_grid, train_df, evaluator, parallelism=4, train_ratio=0.8):
                tvs = TrainValidationSplit(estimator=estimator, estimatorParamMaps=param_grid, evaluator=evaluator,
                    trainRatio=train_ratio, parallelism=parallelism)
                return tvs.fit(train_df).bestModel

            @udf(VectorUDT())
            def to_raw_vec(p):
                p = float(p)
                return Vectors.dense([1.0 - p, p])

            # Spark4-safe probability extraction
            def score_with_p1(model, df, p_col_name):
                return (model.transform(df).select(ID_COL, col(TRAIN_LABEL_COL).cast("int").alias(TRAIN_LABEL_COL),
                                                   vector_to_array(col("probability"))[1].alias(p_col_name)))

            # ============================================================
            # 6) Train/tune base models (Round 0)
            # ============================================================
            lr = SparkLogisticRegression(featuresCol=FEAT_COL, labelCol=TRAIN_LABEL_COL, maxIter=120,
                                         weightCol="class_weight")
            lr_grid = (ParamGridBuilder().addGrid(lr.regParam, [1e-5, 1e-4, 1e-3, 1e-2]).addGrid(lr.elasticNetParam,
                                                                                                 [0.0, 0.5,
                                                                                                  1.0]).build())
            lr_model = tune_model(lr, lr_grid, df_train_w, evaluator_pr)

            rf = SparkRandomForestClassifier(featuresCol=FEAT_COL, labelCol=TRAIN_LABEL_COL, seed=42)
            rf_grid = (ParamGridBuilder().addGrid(rf.numTrees, [200, 400]).addGrid(rf.maxDepth, [6, 10]).build())
            rf_model = tune_model(rf, rf_grid, df_train_clean, evaluator_pr)

            gbt = SparkGBTClassifier(featuresCol=FEAT_COL, labelCol=TRAIN_LABEL_COL, seed=42)
            gbt_grid = (ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5]).addGrid(gbt.maxIter, [60, 120]).build())
            gbt_model = tune_model(gbt, gbt_grid, df_train_clean, evaluator_pr)

            print("[OK] Base models tuned and trained (Round 0).")

            # ============================================================
            # 7) Choose best ensemble weights on VAL (AUC-PR)
            # ============================================================
            def pick_best_weights(df_val_local, lr_m, rf_m, gbt_m):
                v_lr = score_with_p1(lr_m, df_val_local, "p_lr")
                v_rf = score_with_p1(rf_m, df_val_local, "p_rf").select(ID_COL, "p_rf")
                v_gbt = score_with_p1(gbt_m, df_val_local, "p_gbt").select(ID_COL, "p_gbt")
                v_sc = v_lr.join(v_rf, ID_COL, "inner").join(v_gbt, ID_COL, "inner").cache()
                if v_sc.rdd.isEmpty():
                    raise ValueError("val_scored is empty after joins. Check Node_block_id consistency/duplicates.")

                weight_sets = [(0.34, 0.33, 0.33), (0.20, 0.20, 0.60), (0.20, 0.60, 0.20), (0.60, 0.20, 0.20),
                    (1 / 3, 1 / 3, 1 / 3), ]

                best_w, best_aucpr = None, -1.0
                for w_lr, w_rf, w_gbt in weight_sets:
                    tmp = (v_sc.withColumn("p_ens",
                                           w_lr * col("p_lr") + w_rf * col("p_rf") + w_gbt * col("p_gbt")).withColumn(
                        "rawPrediction", to_raw_vec(col("p_ens"))))
                    aucpr = evaluator_pr.evaluate(tmp)
                    if aucpr > best_aucpr:
                        best_aucpr = aucpr
                        best_w = (w_lr, w_rf, w_gbt)

                return v_sc, best_w, best_aucpr

            val_scored, best_w, best_aucpr = pick_best_weights(df_val, lr_model, rf_model, gbt_model)
            W_LR, W_RF, W_GBT = best_w
            print(f"[VAL] Best weights (LR, RF, GBT) = {best_w} | AUC-PR = {best_aucpr:.6f}")

            # ============================================================
            # 8) Threshold selection: maximize F-beta (beta=0.5, precision-focused)
            # ============================================================
            def prf_at_threshold(df_with_p, thr):
                tmp = df_with_p.select(col(TRAIN_LABEL_COL).cast("int").alias("y"),
                    when(col("p_ens") >= lit(thr), 1).otherwise(0).alias("yhat"))
                agg = tmp.agg(F.sum(((col("yhat") == 1) & (col("y") == 1)).cast("int")).alias("tp"),
                    F.sum(((col("yhat") == 1) & (col("y") == 0)).cast("int")).alias("fp"),
                    F.sum(((col("yhat") == 0) & (col("y") == 1)).cast("int")).alias("fn"), ).collect()[0]

                tp = int(agg["tp"]);
                fp = int(agg["fp"]);
                fn = int(agg["fn"])
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                return precision, recall, f1

            def pick_threshold_fbeta(val_scored_local, w_lr, w_rf, w_gbt, beta=0.5):
                thresholds = [i / 1000 for i in range(50, 950)]  # 0.050..0.949 step 0.001
                val_ens = val_scored_local.withColumn("p_ens",
                    w_lr * col("p_lr") + w_rf * col("p_rf") + w_gbt * col("p_gbt")).cache()

                best_thr, best_score = None, -1.0
                best_p = best_r = best_f1 = None

                b2 = beta * beta
                for t in thresholds:
                    p, r, f1v = prf_at_threshold(val_ens, t)
                    fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-9)
                    if fbeta > best_score:
                        best_score = fbeta
                        best_thr = t
                        best_p, best_r, best_f1 = p, r, f1v

                return best_thr, best_p, best_r, best_f1, best_score

            best_thr, p_val, r_val, f1_val, fbeta_val = pick_threshold_fbeta(val_scored, W_LR, W_RF, W_GBT, beta=0.5)
            print(
                f"[VAL] Threshold by F0.5 = {best_thr:.3f} | P={p_val:.4f} R={r_val:.4f} F1={f1_val:.4f} F0.5={fbeta_val:.4f}")

            # ============================================================
            # 9) Self-training (2 iterations, strict confidence): retrain LR + GBT
            # ============================================================
            SELF_TRAIN_ITERS = 2
            P_HI = 0.995
            P_LO = 0.005

            # start from clean set
            df_refined = df_train_clean.cache()

            for it in range(SELF_TRAIN_ITERS):
                print(f"\n[SELF-TRAIN] Iteration {it + 1}/{SELF_TRAIN_ITERS} (P_HI={P_HI}, P_LO={P_LO})")

                # score full raw train (not only clean) to get more candidates
                tr_lr = score_with_p1(lr_model, df_train, "p_lr")
                tr_rf = score_with_p1(rf_model, df_train, "p_rf").select(ID_COL, "p_rf")
                tr_gbt = score_with_p1(gbt_model, df_train, "p_gbt").select(ID_COL, "p_gbt")

                tr_sc = tr_lr.join(tr_rf, ID_COL, "inner").join(tr_gbt, ID_COL, "inner")
                tr_sc = tr_sc.withColumn("p_ens", W_LR * col("p_lr") + W_RF * col("p_rf") + W_GBT * col("p_gbt"))

                # keep only very confident predictions
                tr_conf = (tr_sc.withColumn("y_new",
                                            when(col("p_ens") >= lit(P_HI), lit(1)).when(col("p_ens") <= lit(P_LO),
                                                                                         lit(0)).otherwise(
                                                lit(None)).cast("int")).dropna(subset=["y_new"]).select(ID_COL,
                                                                                                        FEAT_COL,
                                                                                                        col("y_new").alias(
                                                                                                            TRAIN_LABEL_COL)))

                # merge + dedup
                df_refined = (df_refined.unionByName(tr_conf).dropDuplicates([ID_COL]).cache())

                print("[SELF-TRAIN] Refined label distribution:")
                df_refined.groupBy(TRAIN_LABEL_COL).count().orderBy(TRAIN_LABEL_COL).show()

                # weights again
                n_pos_r = df_refined.filter(col(TRAIN_LABEL_COL) == 1).count()
                n_neg_r = df_refined.filter(col(TRAIN_LABEL_COL) == 0).count()
                pos_w_r = min(20.0, float(n_neg_r) / float(n_pos_r))

                df_refined_w = (df_refined.withColumn("class_weight",
                                                      when(col(TRAIN_LABEL_COL) == 1, lit(pos_w_r)).otherwise(
                                                          lit(1.0))).cache())

                # retrain LR + GBT (RF stays fixed)
                lr_model = tune_model(lr, lr_grid, df_refined_w, evaluator_pr)
                gbt_model = tune_model(gbt, gbt_grid, df_refined, evaluator_pr)

                # refresh best weights (AUC-PR) after retraining
                val_scored, best_w, best_aucpr = pick_best_weights(df_val, lr_model, rf_model, gbt_model)
                W_LR, W_RF, W_GBT = best_w
                print(f"[SELF-TRAIN][VAL] Best weights = {best_w} | AUC-PR = {best_aucpr:.6f}")

                # refresh threshold by F0.5
                best_thr, p_val, r_val, f1_val, fbeta_val = pick_threshold_fbeta(val_scored, W_LR, W_RF, W_GBT,
                                                                                 beta=0.5)
                print(
                    f"[SELF-TRAIN][VAL] Threshold(F0.5)={best_thr:.3f} | P={p_val:.4f} R={r_val:.4f} F1={f1_val:.4f} F0.5={fbeta_val:.4f}")

            # ============================================================
            # 10) TEST metrics (counts)
            # ============================================================
            test_lr = score_with_p1(lr_model, df_test, "p_lr")
            test_rf = score_with_p1(rf_model, df_test, "p_rf").select(ID_COL, "p_rf")
            test_gbt = score_with_p1(gbt_model, df_test, "p_gbt").select(ID_COL, "p_gbt")

            test_scored = test_lr.join(test_rf, ID_COL, "inner").join(test_gbt, ID_COL, "inner").cache()
            if test_scored.rdd.isEmpty():
                raise ValueError("test_scored is empty after joins. Check Node_block_id consistency in test.")

            pred_test = (test_scored.withColumn("p_ens", W_LR * col("p_lr") + W_RF * col("p_rf") + W_GBT * col(
                "p_gbt")).withColumn("yhat", when(col("p_ens") >= lit(best_thr), 1).otherwise(0)).select(
                col(TRAIN_LABEL_COL).cast("int").alias("y"), col("yhat").cast("int").alias("yhat")).cache())

            agg = pred_test.agg(F.sum(((col("yhat") == 1) & (col("y") == 1)).cast("int")).alias("tp"),
                F.sum(((col("yhat") == 1) & (col("y") == 0)).cast("int")).alias("fp"),
                F.sum(((col("yhat") == 0) & (col("y") == 1)).cast("int")).alias("fn"),
                F.sum(((col("yhat") == 0) & (col("y") == 0)).cast("int")).alias("tn"), ).collect()[0]

            tp, fp, fn, tn = int(agg["tp"]), int(agg["fp"]), int(agg["fn"]), int(agg["tn"])
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            print("\n=== ENSEMBLE PERFORMANCE ON TEST ===")
            print(f"Precision (anomaly=1): {precision:.4f}")
            print(f"Recall    (anomaly=1): {recall:.4f}")
            print(f"F1-score  (anomaly=1): {f1:.4f}")
            print("\nConfusion Matrix counts:")
            print(f"TP={tp}  FP={fp}")
            print(f"FN={fn}  TN={tn}")

            return {"lr_model": lr_model, "rf_model": rf_model, "gbt_model": gbt_model, "weights": (W_LR, W_RF, W_GBT),
                "threshold": float(best_thr),
                "test_metrics": {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp,
                                 "fp": fp, "fn": fn, "tn": tn}, }


            '''
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
            '''

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
