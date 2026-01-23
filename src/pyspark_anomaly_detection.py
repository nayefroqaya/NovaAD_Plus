import colorama
import numpy as np
import time
import warnings
from pyspark.ml import Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, expr
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.sql.functions import col, when
from scipy.stats import randint, uniform
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sparkxgb import XGBoostClassifier
from pyspark.ml.classification import (
    RandomForestClassifier,
    DecisionTreeClassifier,
    LogisticRegression
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit


warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:

    @staticmethod
    def anomaly_detector(df_final, df_val, df_test,mode):

        train_df = df_final.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                          "features").withColumnRenamed(
            "Final_Label", "label")

        test_df = df_test.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                        "features").withColumnRenamed(
            "Final_Label", "label")
        val_df = df_val.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                        "features").withColumnRenamed(
            "Final_Label", "label")

        # ------------------------------
        # 1. Prepare datasets
        # ------------------------------
        train_df = df_final.select("features_vec_final", "Final_Label") \
            .withColumnRenamed("features_vec_final", "features") \
            .withColumnRenamed("Final_Label", "label")

        val_df = df_val.select("features_vec_final", "Final_Label") \
            .withColumnRenamed("features_vec_final", "features") \
            .withColumnRenamed("Final_Label", "label")

        test_df = df_test.select("features_vec_final", "Final_Label") \
            .withColumnRenamed("features_vec_final", "features") \
            .withColumnRenamed("Final_Label", "label")

        if mode == 'M':
            start_fit = time.time()

            # ------------------------------
            # 2. Class weighting
            # ------------------------------
            label_counts = train_df.groupBy("label").count().collect()
            total_count = sum(r["count"] for r in label_counts)
            class_weights = {r["label"]: total_count / (2.0 * r["count"]) for r in label_counts}

            train_df = train_df.withColumn(
                "class_weight",
                when(col("label") == 0, class_weights[0]).otherwise(class_weights[1])
            )

            # ------------------------------
            # 3. Base Models (RF + LR)
            # ------------------------------
            # Logistic Regression
            lr = LogisticRegression(
                featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="lr_prob", predictionCol="lr_pred", maxIter=60
            )

            # Random Forest
            rf = RandomForestClassifier(
                featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="rf_prob", rawPredictionCol="rf_raw", predictionCol="rf_pred"
            )

            # ------------------------------
            # 4. Tune RF with small TrainValidationSplit
            # ------------------------------
            rf_param_grid = (ParamGridBuilder()
                             .addGrid(rf.numTrees, [150])
                             .addGrid(rf.maxDepth, [18, 24])
                             .addGrid(rf.minInstancesPerNode, [1])
                             .build())

            rf_tvs = TrainValidationSplit(
                estimator=rf,
                estimatorParamMaps=rf_param_grid,
                evaluator=BinaryClassificationEvaluator(
                    labelCol="label", rawPredictionCol="rf_raw", metricName="areaUnderPR"
                ),
                trainRatio=0.8,
                parallelism=4
            )

            rf_tuned_model = rf_tvs.fit(train_df)
            best_rf_model = rf_tuned_model.bestModel

            # ------------------------------
            # 5. Train LR
            # ------------------------------
            lr_model = lr.fit(train_df)

            # ------------------------------
            # 6. Create OOF meta-features (3-fold stacking)
            # ------------------------------
            def create_oof_meta_features(df: DataFrame, folds=1):
                df = df.withColumn("row_idx", lit(0))  # dummy column for splitting
                # Split df into K folds
                fold_size = df.count() // folds
                oof_meta = None

                for k in range(folds):
                    start = k * fold_size
                    end = start + fold_size if k < folds - 1 else None

                    val_fold = df.limit(end).subtract(df.limit(start)) if end else df.limit(df.count()).subtract(df.limit(start))
                    train_fold = df.subtract(val_fold)

                    # Train base models on train_fold
                    lr_fold_model = lr.fit(train_fold)
                    rf_fold_model = best_rf_model  # RF already tuned, just use it

                    # Predict on val_fold
                    def add_probs(df_in, lr_mdl, rf_mdl):
                        df_out = lr_mdl.transform(df_in).withColumn("lr_arr", vector_to_array("lr_prob")) \
                            .withColumn("lr_0", col("lr_arr")[0]).withColumn("lr_1", col("lr_arr")[1])
                        df_out = rf_mdl.transform(df_out).withColumn("rf_arr", vector_to_array("rf_prob")) \
                            .withColumn("rf_0", col("rf_arr")[0]).withColumn("rf_1", col("rf_arr")[1])
                        return df_out

                    val_meta = add_probs(val_fold, lr_fold_model, rf_fold_model)

                    if oof_meta is None:
                        oof_meta = val_meta
                    else:
                        oof_meta = oof_meta.union(val_meta)

                return oof_meta

            train_meta = create_oof_meta_features(train_df, folds=3)

            meta_features = ["lr_0", "lr_1", "rf_0", "rf_1"]
            assembler = VectorAssembler(inputCols=meta_features, outputCol="meta_features")
            train_meta = assembler.transform(train_meta)

            # ------------------------------
            # 7. Meta Logistic Regression tuning
            # ------------------------------
            meta_lr = LogisticRegression(
                featuresCol="meta_features", labelCol="label",
                predictionCol="last_pred_label", probabilityCol="final_prob",
                rawPredictionCol="meta_raw", weightCol="class_weight", maxIter=50
            )

            meta_param_grid = (ParamGridBuilder()
                               .addGrid(meta_lr.regParam, [ 0.01, 0.1])  # 0.0
                               .addGrid(meta_lr.elasticNetParam, [ 0.5, 1.0]) # 0.0
                               .build())

            meta_tvs = TrainValidationSplit(
                estimator=meta_lr,
                estimatorParamMaps=meta_param_grid,
                evaluator=BinaryClassificationEvaluator(
                    labelCol="label", rawPredictionCol="meta_raw", metricName="areaUnderPR"
                ),
                trainRatio=0.8,
                parallelism=2
            )

            stack_model = meta_tvs.fit(train_meta).bestModel

            end_fit = time.time()
            fit_time = (end_fit - start_fit) / 60

            # ------------------------------
            # 8. Validation threshold tuning
            # ------------------------------
            def add_probs(df_in, lr_mdl, rf_mdl):
                df_out = lr_mdl.transform(df_in).withColumn("lr_arr", vector_to_array("lr_prob")) \
                    .withColumn("lr_0", col("lr_arr")[0]).withColumn("lr_1", col("lr_arr")[1])
                df_out = rf_mdl.transform(df_out).withColumn("rf_arr", vector_to_array("rf_prob")) \
                    .withColumn("rf_0", col("rf_arr")[0]).withColumn("rf_1", col("rf_arr")[1])
                return df_out

            val_meta = add_probs(val_df, lr_model, best_rf_model)
            val_meta = assembler.transform(val_meta)
            val_preds = stack_model.transform(val_meta)
            val_preds = val_preds.withColumn("prob_1", vector_to_array("final_prob")[1])

            #thresholds = [i / 100 for i in range(20, 80, 2)]
            thresholds = [i / 100 for i in range(30, 71, 5)]  # 30%, 35%, ..., 70%

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

            print(f"Optimal threshold (validation F1): {best_threshold:.2f}")

            # ------------------------------
            # 9. Test predictions
            # ------------------------------
            start_predict = time.time()
            test_meta = add_probs(test_df, lr_model, best_rf_model)
            test_meta = assembler.transform(test_meta)
            final_test_predictions = (
                stack_model.transform(test_meta)
                .withColumn("prob_1", vector_to_array("final_prob")[1])
                .withColumn("last_pred_label", when(col("prob_1") >= best_threshold, 1).otherwise(0))
            )
            end_predict = time.time()
            predict_time = (end_predict - start_predict) / 60

            # ------------------------------
            # 10. Return results
            # ------------------------------
            return {
                "predictions_df": final_test_predictions,
                "best_threshold": best_threshold,
                "fit_time": fit_time,
                "predict_time": predict_time
            }






        exit()
        if mode == 'M':

            start_fit = time.time()

            # =====================================================
            # 2. Class weighting (imbalanced anomaly detection)
            # =====================================================
            label_counts = train_df.groupBy("label").count().collect()
            total_count = sum(r["count"] for r in label_counts)

            class_weights = {r["label"]: total_count / (2.0 * r["count"]) for r in label_counts}

            train_df = train_df.withColumn("class_weight",
                when(col("label") == 0, class_weights[0]).otherwise(class_weights[1]))

            # =====================================================
            # 3. Base models (level-1)
            # =====================================================
            lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="lr_prob", predictionCol="lr_pred", maxIter=60)

            rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="rf_prob",  # unique
                rawPredictionCol="rf_raw",  # unique
                predictionCol="rf_pred",  # unique
                numTrees=150, maxDepth=24)

            dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="dt_prob",  # unique
                rawPredictionCol="dt_raw",  # unique
                predictionCol="dt_pred",  # unique
                maxDepth=20)

            # =====================================================
            # 4. Train base models
            # =====================================================
            lr_model = lr.fit(train_df)
            rf_model = rf.fit(train_df)
            dt_model = dt.fit(train_df)

            # =====================================================
            # 5. Create meta-features (TRAIN)
            # =====================================================
            def add_probs(df, model, prob_col, prefix):
                return (
                    model.transform(df).withColumn(f"{prefix}_arr", vector_to_array(prob_col)).withColumn(f"{prefix}_0",
                                                                                                          col(f"{prefix}_arr")[
                                                                                                              0]).withColumn(
                        f"{prefix}_1", col(f"{prefix}_arr")[1]))

            train_meta = train_df
            train_meta = add_probs(train_meta, lr_model, "lr_prob", "lr")
            train_meta = add_probs(train_meta, rf_model, "rf_prob", "rf")
            train_meta = add_probs(train_meta, dt_model, "dt_prob", "dt")

            meta_features = ["lr_0", "lr_1", "rf_0", "rf_1", "dt_0", "dt_1"]

            assembler = VectorAssembler(inputCols=meta_features, outputCol="meta_features")

            train_meta = assembler.transform(train_meta)

            # =====================================================
            # 6. Meta-model (stacking)
            # =====================================================
            meta_lr = LogisticRegression(featuresCol="meta_features", labelCol="label",

                predictionCol="last_pred_label",  # final prediction
                probabilityCol="final_prob",  # final probability

                rawPredictionCol="meta_raw",  # ✅ UNIQUE
                weightCol="class_weight", maxIter=50)

            stack_model = meta_lr.fit(train_meta)

            end_fit = time.time()
            fit_time = (end_fit - start_fit) / 60

            # =====================================================
            # 7. VALIDATION — threshold optimization
            # =====================================================
            val_meta = val_df
            val_meta = add_probs(val_meta, lr_model, "lr_prob", "lr")
            val_meta = add_probs(val_meta, rf_model, "rf_prob", "rf")
            val_meta = add_probs(val_meta, dt_model, "dt_prob", "dt")
            val_meta = assembler.transform(val_meta)

            val_preds = stack_model.transform(val_meta)

            val_preds = val_preds.withColumn("prob_1", vector_to_array("final_prob")[1])

            # Grid search for threshold (Spark-safe)
            thresholds = [i / 100 for i in range(5, 95)]
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

            print(f"Optimal threshold (validation F1): {best_threshold:.2f}")

            # =====================================================
            # 8. TEST — final predictions ONLY
            # =====================================================
            start_predict = time.time()

            test_meta = test_df
            test_meta = add_probs(test_meta, lr_model, "lr_prob", "lr")
            test_meta = add_probs(test_meta, rf_model, "rf_prob", "rf")
            test_meta = add_probs(test_meta, dt_model, "dt_prob", "dt")
            test_meta = assembler.transform(test_meta)

            final_test_predictions = (
                stack_model.transform(test_meta).withColumn("prob_1", vector_to_array("final_prob")[1]).withColumn(
                    "last_pred_label", when(col("prob_1") >= best_threshold, 1).otherwise(0)))

            end_predict = time.time()
            predict_time = (end_predict - start_predict) / 60

            # =====================================================
            # 9. RETURN (evaluation happens elsewhere)
            # =====================================================
            return {"predictions_df": final_test_predictions, "best_threshold": best_threshold, "fit_time": fit_time,
                "predict_time": predict_time}

    '''
    def anomaly_detector(df_final,df_val, df_test , X_train, y_train, X_test, y_test_truth, X_val, y_val_truth):
        #X_train, y_train, X_test,
        #y_test_truth, X_val, y_val_truth, mode
        #print("Starting model training process...")
        """
        df_final : PySpark DataFrame (training data)
        df_test  : PySpark DataFrame (test data)

        Required columns:
            features_vec_final : vector
            Final_Label        : integer 0/1

        Output:
            df_test with last_pred_label
            fit_time (minutes)
        """

        # --------------------------------------------
        # 1. Select required columns
        # --------------------------------------------
        train_df = df_final.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                          "features").withColumnRenamed(
            "Final_Label", "label")

        test_df = df_test.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                        "features").withColumnRenamed(
            "Final_Label", "label")
        val_df = df_val.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                        "features").withColumnRenamed(
            "Final_Label", "label")

        label_counts = train_df.groupBy("label").count().collect()
        total_count = sum([row["count"] for row in label_counts])
        class_weights = {row["label"]: total_count / (2 * row["count"]) for row in label_counts}

        train_df = train_df.withColumn("class_weight",
                                       when(col("label") == 0, class_weights[0]).otherwise(class_weights[1]))
        # -------------------------------------------

        # -------------------------------------------------------------------
        # 2. Base Models (level-1)
        # -------------------------------------------------------------------
        # Logistic Regression
        lr = LogisticRegression(featuresCol="features", labelCol="label", probabilityCol="lr_prob",
                                weightCol="class_weight", rawPredictionCol="lr_raw", predictionCol="lr_pred",
                                maxIter=60)

        # Random Forest
        rf = RandomForestClassifier(featuresCol="features", labelCol="label", probabilityCol="rf_prob",
                                    weightCol="class_weight", rawPredictionCol="rf_raw", predictionCol="rf_pred",
                                    numTrees=150, maxDepth=24)

        # Decision Tree (replacing GBT)
        dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", probabilityCol="dt_prob",
                                    weightCol="class_weight", rawPredictionCol="dt_raw", predictionCol="dt_pred",
                                    maxDepth=20)

        # -------------------------------------------------------------------
        # 3. Train base models on train_df
        # -------------------------------------------------------------------
        lr_model = lr.fit(train_df)
        rf_model = rf.fit(train_df)
        dt_model = dt.fit(train_df)

        # -------------------------------------------------------------------
        # 4. Add meta-features
        # -------------------------------------------------------------------
        # LR probabilities
        df1 = (lr_model.transform(train_df).withColumn("lr_prob_arr", vector_to_array("lr_prob")).withColumn("lr_prob0",
                                                                                                             col("lr_prob_arr")[
                                                                                                                 0]).withColumn(
            "lr_prob1", col("lr_prob_arr")[1]))

        # RF probabilities
        df2 = (rf_model.transform(df1).withColumn("rf_prob_arr", vector_to_array("rf_prob")).withColumn("rf_prob0",
                                                                                                        col("rf_prob_arr")[
                                                                                                            0]).withColumn(
            "rf_prob1", col("rf_prob_arr")[1]))

        # DT probabilities
        df3 = (dt_model.transform(df2).withColumn("dt_prob_arr", vector_to_array("dt_prob")).withColumn("dt_prob0",
                                                                                                        col("dt_prob_arr")[
                                                                                                            0]).withColumn(
            "dt_prob1", col("dt_prob_arr")[1]))

        # Meta-feature column names
        meta_features = ["lr_prob0", "lr_prob1", "rf_prob0", "rf_prob1", "dt_prob0", "dt_prob1"]

        # Assemble meta-features
        assembler = VectorAssembler(inputCols=meta_features, outputCol="meta_features")
        train_meta = assembler.transform(df3)

        # -------------------------------------------
        # 4. Train Stacking Meta-model (Logistic Regression)
        # -------------------------------------------
        meta_lr = LogisticRegression(featuresCol="meta_features", labelCol="label", predictionCol="last_pred_label",
                                     probabilityCol="final_prob", weightCol="class_weight")

        stack_model = meta_lr.fit(train_meta)

        # -------------------------------------------
        # 5. Generate Meta-features (test)
        # -------------------------------------------

        # LR
        t1 = (lr_model.transform(test_df).withColumn("lr_prob_arr", vector_to_array("lr_prob")).withColumn("lr_prob0",
                                                                                                           col("lr_prob_arr")[
                                                                                                               0]).withColumn(
            "lr_prob1", col("lr_prob_arr")[1]))

        # RF
        t2 = (rf_model.transform(t1).withColumn("rf_prob_arr", vector_to_array("rf_prob")).withColumn("rf_prob0",
                                                                                                      col("rf_prob_arr")[
                                                                                                          0]).withColumn(
            "rf_prob1", col("rf_prob_arr")[1]))

        # DT
        t3 = (dt_model.transform(t2).withColumn("dt_prob_arr", vector_to_array("dt_prob")).withColumn("dt_prob0",
                                                                                                      col("dt_prob_arr")[
                                                                                                          0]).withColumn(
            "dt_prob1", col("dt_prob_arr")[1]))

        # Assemble meta-features
        test_meta = assembler.transform(t3)

        # -------------------------------------------
        # 6. Apply Stacking Meta-model
        # -------------------------------------------
        final_test_predictions = stack_model.transform(test_meta)

        # -------------------------------------------
        # 7. Evaluate
        # -------------------------------------------

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="last_pred_label",
                                                      metricName="f1")

        f1 = evaluator.evaluate(final_test_predictions)
        print("Stacked Model F1 Score =", f1)

        final_test_predictions.select("label", "last_pred_label", "final_prob").show(20, truncate=False)

        return final_test_predictions, df_final
 '''