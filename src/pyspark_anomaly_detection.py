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

warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:

    @staticmethod
    def anomaly_detector(df_final, df_test, spark):
        print("Starting model training process...")
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
