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

warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:

    @staticmethod
    def anomaly_detector(df_final_train, df_val, df_test, mode):

        df_final_train = df_final_train.select("Node_block_id", "features_vec_final", "Final_Label")
        df_val = df_val.select("Node_block_id", "features_vec_final", "Final_Label")
        df_test = df_test.select("Node_block_id", "features_vec_final", "Final_Label")

        train_df = (df_final_train.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
                                                                                                 "features").withColumnRenamed(
            "Final_Label", "label").cache())

        val_df = (df_val.select("features_vec_final", "Final_Label").withColumnRenamed("features_vec_final",
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

        else:   # -----------------------------------

            # =====================================================
            # 1. AUTOMATIC CLASS WEIGHTS
            # =====================================================

            label_counts = train_df.groupBy("label").count().collect()
            total_count = sum(r["count"] for r in label_counts)

            class_weights = {r["label"]: total_count / (2.0 * r["count"]) for r in label_counts}
            print("Class weights:", class_weights)

            train_df = train_df.withColumn("class_weight",
                when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(class_weights.get(1, 1.0)))

            #val_df = val_df.withColumn("class_weight",
            #    when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(class_weights.get(1, 1.0)))
            #row_count = val_df.count()
            #print(f"\n[DEBUG] DataFrame validation  row count = {row_count}")
            #exit()


            test_df = test_df.withColumn("class_weight",
                when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(class_weights.get(1, 1.0)))
            print("TRAIN COLS:", train_df.columns)

            # =====================================================
            # 2. RANDOM FOREST TRAINING
            # =====================================================
            rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="rf_prob", rawPredictionCol="rf_raw", predictionCol="prediction", numTrees=400,
                maxDepth=25, minInstancesPerNode=5, subsamplingRate=0.8, featureSubsetStrategy="sqrt", seed=42)

            rf_model = rf.fit(train_df)

            # =====================================================
            # 3. PREDICTIONS
            # =====================================================
            #val_preds = rf_model.transform(val_df)
            test_preds = rf_model.transform(test_df)

            # =====================================================
            # 4. SKLEARN-STYLE CLASSIFICATION REPORT
            # =====================================================
            def print_classification_report(df, label_col="label", pred_col="prediction", name=""):
                row_count = df.count()
                print(f"\n[DEBUG] {name} DataFrame row count = {row_count}")

                if row_count == 0:
                    print(f"❌ ERROR: {name} DataFrame is EMPTY.")
                    return

                df2 = df.select(col(pred_col).cast("double").alias("prediction"),
                    col(label_col).cast("double").alias("label")).dropna()

                df2_count = df2.count()
                print(f"[DEBUG] {name} after dropna row count = {df2_count}")

                if df2_count == 0:
                    print(f"❌ ERROR: {name} preds+labels EMPTY after dropna.")
                    df.select(label_col, pred_col).show(20, truncate=False)
                    return

                preds_and_labels = df2.rdd.map(lambda r: (r["prediction"], r["label"]))

                metrics = MulticlassMetrics(preds_and_labels)

                # Version-safe labels from confusion matrix
                cm = metrics.confusionMatrix().toArray()
                num_classes = cm.shape[0]
                labels = list(range(num_classes))

                print(f"\n================ CLASSIFICATION REPORT: {name} ================")
                print(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<10}")

                for lbl in labels:
                    lbl_f = float(lbl)  # ✅ IMPORTANT for old Spark / Py4J

                    precision = metrics.precision(lbl_f)
                    recall = metrics.recall(lbl_f)
                    f1 = metrics.fMeasure(lbl_f)
                    support = int(cm[lbl].sum())

                    print(f"{lbl:<8}{precision:<12.4f}{recall:<12.4f}{f1:<12.4f}{support:<10}")

                print("\nConfusion Matrix (rows=true, cols=pred):")
                print(cm)

                print(f"\nWeighted Precision: {metrics.weightedPrecision():.4f}")
                print(f"Weighted Recall:    {metrics.weightedRecall():.4f}")
                print(f"Weighted F1:        {metrics.weightedFMeasure():.4f}")

            # =====================================================
            # 5. RUN REPORTS
            # =====================================================
            #print_classification_report(val_preds, name="RF Validation")
            #print_classification_report(test_preds, name="RF Test")


            exit()

            # =====================================================
            # 1. AUTOMATIC CLASS WEIGHTS (SAFE)
            # =====================================================
            label_counts = train_df.groupBy("label").count().collect()
            total_count = sum(r["count"] for r in label_counts)

            class_weights = {r["label"]: total_count / (2.0 * r["count"]) for r in label_counts}

            print("Class weights:", class_weights)

            train_df = train_df.withColumn("class_weight", when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(
                class_weights.get(1, 1.0)))

            val_df = val_df.withColumn("class_weight", when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(
                class_weights.get(1, 1.0)))

            test_df = test_df.withColumn("class_weight", when(col("label") == 0, class_weights.get(0, 1.0)).otherwise(
                class_weights.get(1, 1.0)))

            print("TRAIN COLS:", train_df.columns)

            rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="rf_prob", rawPredictionCol="rf_raw", predictionCol="prediction", numTrees=400,
                # try 200–600
                maxDepth=25,  # try 20–30
                minInstancesPerNode=5,  # allow rare patterns
                subsamplingRate=0.8, featureSubsetStrategy="sqrt", seed=42)

            rf_model = rf.fit(train_df)

            val_preds = rf_model.transform(val_df)
            test_preds = rf_model.transform(test_df)

            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

            val_f1 = evaluator.evaluate(val_preds)
            test_f1 = evaluator.evaluate(test_preds)

            print(f"🔥 RF Validation F1:  {val_f1:.4f}")
            print(f"🔥 RF Test F1:        {test_f1:.4f}")

            exit()

            '''

            # =====================================================
            # 2. BASE MODELS
            # =====================================================

            lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="lr_prob", predictionCol="lr_pred", maxIter=150, regParam=0.01, elasticNetParam=0.0)

            rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="class_weight",
                probabilityCol="rf_prob", rawPredictionCol="rf_raw", predictionCol="rf_pred", numTrees=200, maxDepth=20,
                minInstancesPerNode=10, subsamplingRate=0.8, featureSubsetStrategy="sqrt")

            # =====================================================
            # 3. TRAIN BASE MODELS
            # =====================================================

            start_fit = time.time()

            lr_model = lr.fit(train_df)
            rf_model = rf.fit(train_df)

            end_fit = time.time()
            fit_time = (end_fit - start_fit) / 60

            # =====================================================
            # 4. META FEATURE FUNCTIONS
            # =====================================================

            def add_probs(df, model, prob_col, prefix):
                return (
                    model.transform(df).withColumn(f"{prefix}_arr", vector_to_array(prob_col)).withColumn(f"{prefix}_0",
                                                                                                          col(f"{prefix}_arr")[
                                                                                                              0]).withColumn(
                        f"{prefix}_1", col(f"{prefix}_arr")[1]))

            def add_meta_extras(df, prefix):
                df = df.withColumn(f"{prefix}_margin", abs(col(f"{prefix}_1") - col(f"{prefix}_0")))

                df = df.withColumn(f"{prefix}_entropy", -(
                        col(f"{prefix}_1") * log(col(f"{prefix}_1") + 1e-9) + col(f"{prefix}_0") * log(
                    col(f"{prefix}_0") + 1e-9)))
                return df

            # =====================================================
            # 5. TRAIN META
            # =====================================================

            train_meta = train_df

            train_meta = add_probs(train_meta, lr_model, "lr_prob", "lr")
            train_meta = add_meta_extras(train_meta, "lr")

            train_meta = add_probs(train_meta, rf_model, "rf_prob", "rf")
            train_meta = add_meta_extras(train_meta, "rf")

            # =====================================================
            # 6. META ASSEMBLER
            # =====================================================

            meta_features = ["lr_0", "lr_1", "lr_margin", "lr_entropy", "rf_0", "rf_1", "rf_margin", "rf_entropy"]

            assembler = VectorAssembler(inputCols=meta_features, outputCol="meta_features")

            train_meta = assembler.transform(train_meta)

            # =====================================================
            # 7. META MODEL
            # =====================================================

            meta_lr = LogisticRegression(featuresCol="meta_features", labelCol="label", weightCol="class_weight",
                predictionCol="last_pred_label", probabilityCol="final_prob", rawPredictionCol="meta_raw", maxIter=100,
                regParam=0.01, elasticNetParam=0.0)

            stack_model = meta_lr.fit(train_meta)

            # =====================================================
            # 8. VALIDATION
            # =====================================================

            val_meta = val_df

            val_meta = add_probs(val_meta, lr_model, "lr_prob", "lr")
            val_meta = add_meta_extras(val_meta, "lr")

            val_meta = add_probs(val_meta, rf_model, "rf_prob", "rf")
            val_meta = add_meta_extras(val_meta, "rf")

            val_meta = assembler.transform(val_meta)

            val_preds = stack_model.transform(val_meta)
            val_preds = val_preds.withColumn("prob_1", vector_to_array("final_prob")[1])

            # =====================================================
            # 9. FAST THRESHOLD TUNING
            # =====================================================

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

            # =====================================================
            # 10. TEST — FINAL PREDICTIONS
            # =====================================================

            start_predict = time.time()

            test_meta = test_df

            test_meta = add_probs(test_meta, lr_model, "lr_prob", "lr")
            test_meta = add_meta_extras(test_meta, "lr")

            test_meta = add_probs(test_meta, rf_model, "rf_prob", "rf")
            test_meta = add_meta_extras(test_meta, "rf")

            test_meta = assembler.transform(test_meta)

            final_test_predictions = (
                stack_model.transform(test_meta).withColumn("prob_1", vector_to_array("final_prob")[1]).withColumn(
                    "last_pred_label", when(col("prob_1") >= best_threshold, 1).otherwise(0)))

            end_predict = time.time()
            predict_time = (end_predict - start_predict) / 60

            # =====================================================
            # 11. RETURN
            # =====================================================

            return {"predictions_df": final_test_predictions, "best_threshold": best_threshold, "fit_time": fit_time,
                "predict_time": predict_time}
            
            '''


            exit()

























            # ------- it is ok but not perfect :
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
