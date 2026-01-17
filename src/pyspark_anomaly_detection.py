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




        if mode != "M":
            return None

        # =====================================================
        # 1. Prepare datasets
        # =====================================================
        def prep(df):
            return (
                df.select("features_vec_final", "Final_Label")
                  .withColumnRenamed("features_vec_final", "features")
                  .withColumnRenamed("Final_Label", "label")
            )

        train_df = prep(df_final)
        val_df   = prep(df_val)
        test_df  = prep(df_test)

        train_df.persist(StorageLevel.MEMORY_AND_DISK)
        val_df.persist(StorageLevel.MEMORY_AND_DISK)
        test_df.persist(StorageLevel.MEMORY_AND_DISK)

        train_count = train_df.count()
        val_df.count()
        test_df.count()

        # =====================================================
        # 2. Adaptive tuning policy (based on data size)
        # =====================================================
        if train_count < 100_000:
            rf_depths = [8, 12]
            rf_trees = 50
            dt_depths = [6, 10]
            lr_regs = [0.0, 0.01]
        elif train_count < 1_000_000:
            rf_depths = [10, 14]
            rf_trees = 75
            dt_depths = [8, 12]
            lr_regs = [0.01, 0.1]
        else:
            rf_depths = [12, 16]
            rf_trees = 100
            dt_depths = [10, 14]
            lr_regs = [0.1]

        # =====================================================
        # 3. Class weights
        # =====================================================
        counts = train_df.groupBy("label").count().collect()
        total = sum(r["count"] for r in counts)
        class_weights = {r["label"]: total / (2.0 * r["count"]) for r in counts}

        train_df = train_df.withColumn(
            "class_weight",
            when(col("label") == 0, class_weights.get(0, 1.0))
            .otherwise(class_weights.get(1, 1.0))
        )

        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            metricName="areaUnderROC"
        )

        start_fit = time.time()

        # =====================================================
        # 4. Logistic Regression (tuned)
        # =====================================================
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            probabilityCol="lr_prob",
            maxIter=30
        )

        lr_grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, lr_regs)
            .addGrid(lr.elasticNetParam, [0.0, 0.5])
            .build()
        )

        lr_tvs = TrainValidationSplit(
            estimator=lr,
            estimatorParamMaps=lr_grid,
            evaluator=evaluator,
            trainRatio=0.8,
            parallelism=4
        )

        lr_model = lr_tvs.fit(train_df).bestModel

        # =====================================================
        # 5. Random Forest (light adaptive tuning)
        # =====================================================
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            probabilityCol="rf_prob",
            rawPredictionCol="rf_raw",
            predictionCol="rf_pred",
            numTrees=rf_trees,
            subsamplingRate=0.7,
            featureSubsetStrategy="sqrt"
        )

        rf_grid = (
            ParamGridBuilder()
            .addGrid(rf.maxDepth, rf_depths)
            .build()
        )

        rf_tvs = TrainValidationSplit(
            estimator=rf,
            estimatorParamMaps=rf_grid,
            evaluator=evaluator,
            trainRatio=0.8,
            parallelism=4
        )

        rf_model = rf_tvs.fit(train_df).bestModel

        # =====================================================
        # 6. Decision Tree (cheap tuning)
        # =====================================================
        dt = DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            probabilityCol="dt_prob",
            rawPredictionCol="dt_raw",
            predictionCol="dt_pred"
        )

        dt_grid = (
            ParamGridBuilder()
            .addGrid(dt.maxDepth, dt_depths)
            .addGrid(dt.minInstancesPerNode, [10, 30])
            .build()
        )

        dt_tvs = TrainValidationSplit(
            estimator=dt,
            estimatorParamMaps=dt_grid,
            evaluator=evaluator,
            trainRatio=0.8,
            parallelism=4
        )

        dt_model = dt_tvs.fit(train_df).bestModel

        # =====================================================
        # 7. Build meta-features (single-pass transforms)
        # =====================================================
        def build_meta(df):
            return (
                lr_model.transform(df)
                .transform(rf_model)
                .transform(dt_model)
                .withColumn("lr_1", vector_to_array("lr_prob")[1])
                .withColumn("rf_1", vector_to_array("rf_prob")[1])
                .withColumn("dt_1", vector_to_array("dt_prob")[1])
                .select("label", "class_weight", "lr_1", "rf_1", "dt_1")
            )

        train_meta = build_meta(train_df).persist(StorageLevel.MEMORY_AND_DISK)
        train_meta.count()

        assembler = VectorAssembler(
            inputCols=["lr_1", "rf_1", "dt_1"],
            outputCol="meta_features"
        )

        train_meta = assembler.transform(train_meta)

        # =====================================================
        # 8. Meta-model (kept simple on purpose)
        # =====================================================
        meta_lr = LogisticRegression(
            featuresCol="meta_features",
            labelCol="label",
            weightCol="class_weight",
            probabilityCol="final_prob",
            predictionCol="last_pred_label",
            maxIter=20
        )

        stack_model = meta_lr.fit(train_meta)

        fit_time = (time.time() - start_fit) / 60

        # =====================================================
        # 9. Validation — vectorized threshold tuning
        # =====================================================
        val_meta = assembler.transform(build_meta(val_df))

        val_preds = (
            stack_model.transform(val_meta)
            .withColumn("prob_1", vector_to_array("final_prob")[1])
            .select("label", "prob_1")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        val_preds.count()

        thresholds_df = val_preds.sparkSession.createDataFrame(
            [(i / 100,) for i in range(5, 95)], ["threshold"]
        )

        metrics = (
            val_preds.crossJoin(thresholds_df)
            .withColumn("pred", col("prob_1") >= col("threshold"))
            .groupBy("threshold")
            .agg(
                expr("sum(case when label=1 and pred then 1 else 0 end)").alias("tp"),
                expr("sum(case when label=0 and pred then 1 else 0 end)").alias("fp"),
                expr("sum(case when label=1 and not pred then 1 else 0 end)").alias("fn"),
            )
            .withColumn("precision", expr("tp / (tp + fp + 1e-6)"))
            .withColumn("recall", expr("tp / (tp + fn + 1e-6)"))
            .withColumn(
                "f1",
                expr("2 * precision * recall / (precision + recall + 1e-6)")
            )
        )

        best_threshold = metrics.orderBy(expr("f1 desc")).first()["threshold"]

        # =====================================================
        # 10. TEST — final predictions
        # =====================================================
        start_predict = time.time()

        test_meta = assembler.transform(build_meta(test_df))

        final_test_predictions = (
            stack_model.transform(test_meta)
            .withColumn("prob_1", vector_to_array("final_prob")[1])
            .withColumn(
                "last_pred_label",
                when(col("prob_1") >= best_threshold, 1).otherwise(0)
            )
        )

        predict_time = (time.time() - start_predict) / 60

        # =====================================================
        # 11. RETURN
        # =====================================================
        return {
            "predictions_df": final_test_predictions,
            "best_threshold": float(best_threshold),
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