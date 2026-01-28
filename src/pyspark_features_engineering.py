import colorama
import math
import numpy as np
import pandas as pd
import warnings
from itertools import product
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.feature import RobustScaler
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import VectorSizeHint, VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.ml.functions import vector_to_array
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import array, col, concat
from pyspark.sql.functions import array_max, col, when
from pyspark.sql.functions import col, avg, when
from pyspark.sql.functions import col, size, max as spark_max
from pyspark.sql.functions import col, sum as spark_sum
from pyspark.sql.functions import col, udf
from pyspark.sql.functions import col, when
from pyspark.sql.functions import col, when, array
from pyspark.sql.functions import col, when, avg, udf
from pyspark.sql.functions import col, when, count, trim
from pyspark.sql.functions import col, when, udf
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import udf
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import *
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import FloatType
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, DoubleType
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class FeaturesEngineering:

    @staticmethod
    def features_aggregation_transformation(final_train_with_test_with_val, dataset):

        print("---- Starting feature aggregation and transformation ----")
        final_train_with_test_with_val.printSchema()
        #        exit()
        # 2️⃣ Drop rows that are fully null
        final_train_with_test_with_val = final_train_with_test_with_val.na.drop(how='all')
        # Sentiment features : -------------------------------------------------
        final_train_with_test_with_val = final_train_with_test_with_val.withColumn("sentiment_label",
                                                                 when(col("sentiment_label").isNull(),
                                                                      "negative")  # placeholder for missing sentiment
                                                                 .otherwise(col("sentiment_label")))

        final_train_with_test_with_val.select(count(when(col("sentiment_label").isNull(), True)).alias("null_count"),
                                     count(when(col("sentiment_label").isNotNull(), True)).alias("not_null_count"))

        print("Columns:", final_train_with_test_with_val.columns)
        final_train_with_test_with_val.select("sentiment_label").printSchema()

        final_train_with_test_with_val = final_train_with_test_with_val.withColumn("sentiment_label", when(
            col("sentiment_label").isNull() | (trim(col("sentiment_label")) == "") | (
                col("sentiment_label").isin("None", "NaN", "null")), "negative").otherwise(col("sentiment_label")))

        # Normalize and clean first (optional but safer)
        final_train_with_test_with_val = final_train_with_test_with_val.withColumn("sentiment_label", trim(col("sentiment_label")))

        # Then map to numeric
        final_train_with_test_with_val = final_train_with_test_with_val.withColumn("sentiment_label_indexed",
                                                                 when(col("sentiment_label") == "positive",
                                                                      1).otherwise(0))

        feature_columns = ["sentiment_label_indexed", "Dominant_Topic", "num_words", "Character_Count", "entropy",
                           "month", "day", "hour", "minute", "second"]

        df_cached = final_train_with_test_with_val
        print(' preparing the numeric array ----')
        df_cached = df_cached.withColumn("numeric_array", array(*[col(c).cast("double") for c in feature_columns]))
        print('preparing  full vector by concating ')

        df_cached = df_cached.withColumn("features_array", concat(col("numeric_array"), col("reduced_embedding")))

        df_cached = df_cached.withColumn("features", array_to_vector("features_array"))
        df_cached.printSchema()
        print('xxxx-------------------')

        df_cached.groupBy("Temp_label").count().show()
        df_cached.groupBy("Label").count().show()
        #        exit()

        log_normal_labelled = df_cached.filter(F.col("Temp_label") == 0)
        log_remain_normal_anomaly_unlabelled = df_cached.filter(F.col("Temp_label") == 999)
        log_test_unlabelled = df_cached.filter(F.col("Temp_label") == 888)
        log_val_labelled = df_cached.filter(F.col("Temp_label") == 777)

        # ------------------ Helper UDFs ----------------------------------------------

        # UDF to average a list of vectors (list of lists)
        @F.udf("array<float>")
        def avg_vector_udf(list_of_vectors):
            import numpy as np
            arr = np.array(list_of_vectors, dtype="float32")
            return arr.mean(axis=0).tolist()

        # ------------------ (1) Train - Normal logs labelled :Temp_label =0 -------------------------
        summed_df_normal_labelled_train = (log_normal_labelled.groupby("Node_block_id").agg(
            avg_vector_udf(F.collect_list("features")).alias("features")))

        sequence_labels_normal_labelled = (log_normal_labelled.groupby("Node_block_id").agg(
            F.when(F.sum(F.when(F.col("Label") != "Normal", 1).otherwise(0)) > 0, "anomaly").otherwise("normal").alias(
                "Label")))

        summed_df_normal_labelled_train = (
            summed_df_normal_labelled_train.join(sequence_labels_normal_labelled, "Node_block_id", "inner").withColumn(
                "Temp_label", F.lit(0)))

        # Validation checks (NOTE: Spark can't exit(), but you can raise exceptions)
        xx_anomaly = summed_df_normal_labelled_train.filter(F.col("Label") == "anomaly")
        yy_normal = summed_df_normal_labelled_train.filter(F.col("Label") == "normal")

        if xx_anomaly.count() != 0 or yy_normal.count() == 0:
            raise Exception("Error: Normal data validation failed")

        if yy_normal.count() != summed_df_normal_labelled_train.count():
            raise Exception("Error: Length mismatch in normal labelled data")

        # ------------------ (2) Train - Unlabelled logs (Temp_label = 999) -----------
        summed_df_combine_unlabelled_train = (log_remain_normal_anomaly_unlabelled.groupby("Node_block_id").agg(
            avg_vector_udf(F.collect_list("features")).alias("features")))

        sequence_labels_combine_unlabelled = (log_remain_normal_anomaly_unlabelled.groupby("Node_block_id").agg(
            F.when(F.sum(F.when(F.col("Label") != "Normal", 1).otherwise(0)) > 0, "anomaly").otherwise("normal").alias(
                "Label")))

        summed_df_combine_unlabelled_train = (
            summed_df_combine_unlabelled_train.join(sequence_labels_combine_unlabelled, "Node_block_id",
                                                    "inner").withColumn("Temp_label", F.lit(999)))

        # Validation checks
        xx_anomaly = summed_df_combine_unlabelled_train.filter(F.col("Label") == "anomaly")
        yy_normal = summed_df_combine_unlabelled_train.filter(F.col("Label") == "normal")

        if xx_anomaly.count() == 0 or yy_normal.count() == 0:
            raise Exception("Error: Unlabeled data validation failed")

        summed_df_train = summed_df_normal_labelled_train.unionByName(summed_df_combine_unlabelled_train)

        # ------------------ (3) Test dataset (Temp_label = 888) -----------------------
        summed_df_combine_unlabelled_test = (log_test_unlabelled.groupby("Node_block_id").agg(
            avg_vector_udf(F.collect_list("features")).alias("features")))

        sequence_labels_combine_unlabelled_test = (log_test_unlabelled.groupby("Node_block_id").agg(
            F.when(F.sum(F.when(F.col("Label") != "Normal", 1).otherwise(0)) > 0, "anomaly").otherwise("normal").alias(
                "Label")))

        summed_df_combine_unlabelled_test = (
            summed_df_combine_unlabelled_test.join(sequence_labels_combine_unlabelled_test, "Node_block_id",
                                                   "inner").withColumn("Temp_label", F.lit(888)))

        summed_df_test = summed_df_combine_unlabelled_test


        # ------------------ (4) validate dataset (Temp_label = 777) -----------------------
        summed_df_combine_labelled_val = (log_val_labelled.groupby("Node_block_id").agg(
            avg_vector_udf(F.collect_list("features")).alias("features")))

        sequence_labels_combine_labelled_val= (log_val_labelled.groupby("Node_block_id").agg(
            F.when(F.sum(F.when(F.col("Label") != "Normal", 1).otherwise(0)) > 0, "anomaly").otherwise("normal").alias(
                "Label")))

        summed_df_combine_labelled_val = (
            summed_df_combine_labelled_val.join(sequence_labels_combine_labelled_val, "Node_block_id",
                                                   "inner").withColumn("Temp_label", F.lit(888)))

        summed_df_val = summed_df_combine_labelled_val

        # ------------------ Final Combine -------------------------------------------

        summ_train_test_val_combine = summed_df_train.unionByName(summed_df_test).unionByName(summed_df_val)
        summ_train_test_val_combine.printSchema()
        print("Feature aggregation completed successfully")
        #        exit()

        # =============================
        print('# 3️⃣ Apply RobustScaler-------')
        # =============================

        array_to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
        summ_train_test_val_combine = summ_train_test_val_combine.withColumn("features_vec",
                                                                     array_to_vector_udf(col("features")))

        scaler = StandardScaler(inputCol="features_vec", outputCol="features_vec_final", withMean=True, withStd=True)
        model = scaler.fit(summ_train_test_val_combine)
        summ_train_test_val_combine_scaled = model.transform(summ_train_test_val_combine)
        print("✅ StandardScaler applied successfully")
        print('scaling done finally -------')

        X_sequences_df = summ_train_test_val_combine_scaled.select("features_vec_final")
        y_sequences_df = summ_train_test_val_combine_scaled.select("Label")

        return summ_train_test_val_combine_scaled, X_sequences_df, y_sequences_df

    @staticmethod
    def novelty_detection_label_establishment(sequences_df: DataFrame, method: str = "gmm"
                                              # options: "IsolationForest" or "rf"
                                              ):
        sequences_df.printSchema()
        sequences_df.groupBy("Label").count().show()
        sequences_df.groupBy("Temp_label").count().show()
        sequences_df = sequences_df.withColumn("Label",
                                               F.when(F.col("Label") == "normal", 0).when(F.col("Label") == "anomaly",
                                                                                          1))
        """
            Novelty detection pipeline supporting both IsolationForest (iof) and RandomForest (rf).

            Returns:
                X_train, y_train, X_test, y_test_truth
            """
        print(f"\n🚀 Starting novelty detection using method = {method.upper()}")

        # ---------------------------------------------------------
        # 1️⃣ Split data
        # ---------------------------------------------------------
        train_normal_df = sequences_df.filter(col("Temp_label") == 0)
        unlabeled_train_df = sequences_df.filter(col("Temp_label") == 999)
        test_df = sequences_df.filter(col("Temp_label") == 888)

        #train_normal_df.select("Temp_label", "Label").show(3, truncate=False)
        #unlabeled_train_df.select("Temp_label", "Label").show(3, truncate=False)
        #test_df.select("Temp_label", "Label").show(3, truncate=False)
        #        exit()

        print('info labe  ..........................................')
        train_normal_df.groupBy("Label").count().show()
        unlabeled_train_df.groupBy("Label").count().show()
        test_df.groupBy("Label").count().show()
        #        exit()
        if train_normal_df.count() == 0:
            raise ValueError("❌ No normal logs (Temp_label=0) found for training.")
        if unlabeled_train_df.count() == 0:
            raise ValueError("❌ No unlabeled logs (Temp_label=999) found for novelty detection.")

        # Ensure feature column is vector type
        if method.lower() == "gmm":

            print("\n☁️ Using Gaussian Mixture Model (semi-supervised) ...")
            # Train on normal logs only
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for GMM novelty detection.")

            feature_col = "features_vec_final"

            # GMM hyperparameter grid
            k_values = [2,3,5]  # [2,3,5,7,9,11]  # number of mixture components
            max_iter_values =  [5,10, 50, 100,150]
            best_model, best_score, best_params = None, -np.inf, None

            for k, max_iter in product(k_values, max_iter_values):
                print(f"🔍 Testing GMM: k={k}, maxIter={max_iter}")
                try:
                    gmm = GaussianMixture(featuresCol=feature_col, predictionCol="gmm_pred",
                                          probabilityCol="probability", k=k, maxIter=max_iter, seed=42)
                    model = gmm.fit(train_normal_df)

                    # Compute mean log-likelihood on unlabeled data
                    preds = model.transform(unlabeled_df)
                    mean_ll = preds.select("probability").rdd.map(lambda x: float(x[0][0])).mean()

                    if mean_ll > best_score:
                        best_score = mean_ll
                        best_params = (k, max_iter)
                        best_model = model
                except Exception as e:
                    print(f"⚠️ Failed for GMM params ({k}, {max_iter}): {e}")
                    continue

            if best_model is None:
                raise RuntimeError("❌ No valid GMM model found.")

            print(f"\n✅ Optimal GMM parameters: k={best_params[0]}, maxIter={best_params[1]}")
            print(f"   Best mean log-likelihood: {best_score:.6f}")
            # Assign pseudo-labels based on likelihood
            preds = best_model.transform(unlabeled_df)

            preds = preds.withColumn("prob_array", vector_to_array("probability"))
            preds = preds.withColumn("anomaly_score", 1 - array_max(col("prob_array")))

            threshold = 0.1  # float(np.percentile(scores, 90))
            pseudo_labels_df = preds.withColumn("pseudo_label", when(col("anomaly_score") > threshold, 1).otherwise(0))

            unlabeled_eval_df = pseudo_labels_df.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")  # rename to avoid ambiguity
                                    ), on="Node_block_id", how="inner")

            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label").toPandas()
            y_true = pdf_unlabeled["true_label"]
            y_pred = pdf_unlabeled["pseudo_label"]
            print('Classification_report only for pseudo-code label for unlabeled data')
            print(classification_report(y_true, y_pred, digits=3))
            #            exit()

            # Merge pseudo-labeled + normal logs
            df_normal = train_normal_df.withColumn("Final_Label", when(col("Temp_label") == 0, 0))
            df_unlabeled = pseudo_labels_df.withColumnRenamed("pseudo_label", "Final_Label")
            df_final_train = df_normal.unionByName(df_unlabeled, allowMissingColumns=True)
            df_test = (sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label")))
            df_val = (sequences_df.filter(col("Temp_label") == 777).withColumn("Final_Label", col("Label")))

            df_final_train.printSchema()
            df_test.printSchema()
            df_val.printSchema()

            # Convert to Pandas
            pdf_final = df_final_train.toPandas()
            y_train = pdf_final["Final_Label"].values
            y_train_truth = pdf_final["Label"].values
            print('Classification_report full training data')
            print(classification_report(y_train_truth, y_train, digits=3))
            #exit()

            # Prepare test set
            df_test = (sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label")))
            pdf_test = df_test.toPandas()

            # Prepare val set
            df_val = (sequences_df.filter(col("Temp_label") == 777).withColumn("Final_Label", col("Label")))
            pdf_val = df_val.toPandas()

            X_train = pdf_final["features_vec_final"].tolist()
            y_train = pdf_final["Final_Label"].values
            X_test = pdf_test["features_vec_final"].tolist()
            y_test_truth = pdf_test["Final_Label"].values

            X_val = pdf_val["features_vec_final"].tolist()
            y_val_truth = pdf_val["Final_Label"].values

            print(f"\n✅ Novelty detection (GMM) completed successfully.")
            #exit()

            return df_final_train, df_test ,df_val  #, X_train, y_train, X_test, y_test_truth, X_val, y_val_truth

