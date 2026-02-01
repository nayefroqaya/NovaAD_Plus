import colorama
import math
import numpy as np
import pandas as pd
import warnings
from kneed import KneeLocator
import numpy as np
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
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, when, array_max
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
import numpy as np
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np
from pyspark.ml.feature import PCA
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import DoubleType
import numpy as np
from scipy.stats import chi2
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
    def novelty_detection_label_establishment(sequences_df: DataFrame,    spark: SparkSession, method: str = "gmm"
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
        #--------------------------------------------------------------------------------
        # Ensure feature column is vector type
        if method.lower() == "gmm":

            '''
            print("\n🧠 Using PCA + KMeans for novelty detection (robust, semi-supervised) ...")

            # -----------------------------
            # 0. Filter normal and unlabeled logs
            # -----------------------------
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for PCA + KMeans novelty detection.")

            feature_col = "features_vec_final"

            # -----------------------------
            # 1. Fit PCA on normal logs
            # -----------------------------
            pca_probe = PCA(k=min(100, len(train_normal_df.columns)), inputCol=feature_col, outputCol="pca_tmp")
            pca_probe_model = pca_probe.fit(train_normal_df)

            explained = np.array(pca_probe_model.explainedVariance.toArray())
            cum_energy = np.cumsum(explained)
            target_energy = 0.95  # 🔧 tune: 0.90, 0.95, 0.99
            k_opt = int(np.searchsorted(cum_energy, target_energy) + 1)
            print(f"📊 PCA target energy={target_energy}, optimal k={k_opt}")

            # Final PCA
            pca = PCA(k=k_opt, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(train_normal_df)

            train_pca = pca_model.transform(train_normal_df)
            unlabeled_pca = pca_model.transform(unlabeled_df)

            # -----------------------------
            # 2. Fit KMeans on PCA features of normal logs
            # -----------------------------
            n_clusters = min(5, train_normal_df.count())  # 🔧 tune: 2,3,5,7
            kmeans = KMeans(featuresCol="pca_features", predictionCol="cluster", k=n_clusters, seed=42)
            kmeans_model = kmeans.fit(train_pca)

            # -----------------------------
            # 3. Compute distance to nearest centroid
            # -----------------------------
            centers = np.array(kmeans_model.clusterCenters())
            bc_centers = spark.sparkContext.broadcast(centers)

            @udf(DoubleType())
            def dist_to_nearest_centroid(vec):
                x = np.array(vec.toArray())
                dists = np.linalg.norm(bc_centers.value - x, axis=1)
                return float(np.min(dists))

            train_pca = train_pca.withColumn("anomaly_score", dist_to_nearest_centroid(col("pca_features")))
            unlabeled_pca = unlabeled_pca.withColumn("anomaly_score", dist_to_nearest_centroid(col("pca_features")))

            # -----------------------------
            # 4. Threshold selection (contamination)
            # -----------------------------
            expected_anomaly_rate = 0.1  # 🔧 tune: 0.01, 0.03, 0.05
            threshold = train_pca.approxQuantile("anomaly_score", [1 - expected_anomaly_rate], 0.01)[0]
            print(f"📏 PCA + KMeans threshold (contamination={expected_anomaly_rate * 100:.1f}%): {threshold:.6f}")

            # -----------------------------
            # 5. Two-zone pseudo-labeling
            # -----------------------------
            low_thr = train_pca.approxQuantile("anomaly_score", [0.90], 0.01)[0]
            pseudo_labels_df = unlabeled_pca.withColumn("pseudo_label",
                when(col("anomaly_score") >= threshold, 1).when(col("anomaly_score") <= low_thr, 0).otherwise(None))

            # Drop uncertain samples
            pseudo_labels_df = pseudo_labels_df.filter(col("pseudo_label").isNotNull())

            # -----------------------------
            # 6. Evaluate pseudo-labels
            # -----------------------------
            unlabeled_eval_df = pseudo_labels_df.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")), on="Node_block_id",
                how="inner")
            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label").toPandas()
            y_true = pdf_unlabeled["true_label"]
            y_pred = pdf_unlabeled["pseudo_label"]

            print('Classification_report pseudo-labels (PCA + KMeans)')
            print(classification_report(y_true, y_pred, digits=3))

            # -----------------------------
            # 7. Merge pseudo-labeled + normal logs
            # -----------------------------
            df_normal = train_normal_df.withColumn("Final_Label", when(col("Temp_label") == 0, 0))
            df_unlabeled = pseudo_labels_df.withColumnRenamed("pseudo_label", "Final_Label")
            df_final_train = df_normal.unionByName(df_unlabeled, allowMissingColumns=True)
            df_test = sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label"))

            df_final_train.printSchema()
            df_test.printSchema()

            # -----------------------------
            # 8. Full training label quality
            # -----------------------------
            pdf_final = df_final_train.toPandas()
            y_train = pdf_final["Final_Label"].values
            y_train_truth = pdf_final["Label"].values

            print('Classification_report full training data (PCA + KMeans novelty)')
            print(classification_report(y_train_truth, y_train, digits=3))
            '''

            '''
            print("\n🧠 Using PCA + Mahalanobis for robust semi-supervised novelty detection ...")

            # -----------------------------
            # 0. Prepare datasets
            # -----------------------------
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for PCA novelty detection.")

            feature_col = "features_vec_final"

            # -----------------------------
            # 1. Probe PCA to get energy curve
            # -----------------------------
            pca_probe = PCA(k=min(60, train_normal_df.select(feature_col).first()[0].size), inputCol=feature_col,
                            outputCol="pca_tmp")
            pca_probe_model = pca_probe.fit(train_normal_df)

            explained = np.array(pca_probe_model.explainedVariance.toArray())
            cum_energy = np.cumsum(explained)

            target_energy = 0.95  # 🔧 tune 0.90,0.95,0.99
            k_opt = int(np.searchsorted(cum_energy, target_energy) + 1)
            print(f"📊 PCA target energy={target_energy}, optimal k={k_opt}")

            # -----------------------------
            # 2. Fit final PCA
            # -----------------------------
            pca = PCA(k=k_opt, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(train_normal_df)

            train_pca = pca_model.transform(train_normal_df)
            unlabeled_pca = pca_model.transform(unlabeled_df)

            # -----------------------------
            # 3. Mahalanobis distance in PCA space
            # -----------------------------
            pdf_train_pca = train_pca.select("pca_features").toPandas()
            Z = np.vstack(pdf_train_pca["pca_features"].apply(lambda v: v.toArray()))
            mu = Z.mean(axis=0)
            cov = np.cov(Z, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6  # regularization
            inv_cov = np.linalg.inv(cov)

            bc_mu = spark.sparkContext.broadcast(mu)
            bc_inv_cov = spark.sparkContext.broadcast(inv_cov)

            @udf(DoubleType())
            def mahalanobis_score(pca_vec):
                z = np.array(pca_vec.toArray())
                d = z - bc_mu.value
                return float(np.sqrt(d.T @ bc_inv_cov.value @ d))

            train_pca = train_pca.withColumn("anomaly_score", mahalanobis_score(col("pca_features")))
            unlabeled_pca = unlabeled_pca.withColumn("anomaly_score", mahalanobis_score(col("pca_features")))

            # -----------------------------
            # 4. Statistical threshold (Chi-square)
            # -----------------------------
            alpha = 0.99  # 🔧 tune 0.99, 0.995, 0.999
            threshold = float(np.sqrt(chi2.ppf(alpha, k_opt)))
            print(f"📏 PCA-Mahalanobis threshold (Chi2, alpha={alpha}): {threshold:.6f}")

            # -----------------------------
            # 5. Two-zone pseudo-labeling (optional)
            # -----------------------------
            low_thr = train_pca.approxQuantile("anomaly_score", [0.80], 0.01)[0]
            print(f"📏 Two-zone thresholds: low={low_thr:.6f}, high={threshold:.6f}")

            pseudo_labels_df = unlabeled_pca.withColumn("pseudo_label",
                when(col("anomaly_score") >= threshold, 1).when(col("anomaly_score") <= low_thr, 0).otherwise(None))

            # Drop uncertain samples
            pseudo_labels_df = pseudo_labels_df.filter(col("pseudo_label").isNotNull())

            # -----------------------------
            # 6. Evaluate pseudo-labels (optional)
            # -----------------------------
            unlabeled_eval_df = pseudo_labels_df.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")), on="Node_block_id",
                how="inner")

            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label").toPandas()
            y_true = pdf_unlabeled["true_label"]
            y_pred = pdf_unlabeled["pseudo_label"]
            print('Classification_report on pseudo-labels (PCA-Mahalanobis)')
            print(classification_report(y_true, y_pred, digits=3))

            # -----------------------------
            # 7. Merge pseudo-labeled + normal
            # -----------------------------
            df_normal = train_normal_df.withColumn("Final_Label", when(col("Temp_label") == 0, 0))
            df_unlabeled = pseudo_labels_df.withColumnRenamed("pseudo_label", "Final_Label")

            df_final_train = df_normal.unionByName(df_unlabeled, allowMissingColumns=True)
            df_test = sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label"))

            df_final_train.printSchema()
            df_test.printSchema()

            # -----------------------------
            # 8. Full training label quality (diagnostic)
            # -----------------------------
            pdf_final = df_final_train.toPandas()
            y_train = pdf_final["Final_Label"].values
            y_train_truth = pdf_final["Label"].values

            print('Classification_report full training data (PCA-Mahalanobis)')
            print(classification_report(y_train_truth, y_train, digits=3))
            '''
            '''
            print("\n🧠 Using PCA for novelty detection (robust, semi-supervised) ...")

            # Train on normal logs only
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for PCA novelty detection.")

            feature_col = "features_vec_final"

            # -----------------------------
            # 1. First PCA to get energy curve
            # -----------------------------
            pca_probe = PCA(k=59, inputCol=feature_col, outputCol="pca_tmp")
            pca_probe_model = pca_probe.fit(train_normal_df)

            explained = np.array(pca_probe_model.explainedVariance.toArray())
            cum_energy = np.cumsum(explained)

            target_energy = 0.95  # 🔧 try 0.90, 0.95, 0.99
            k_opt = int(np.searchsorted(cum_energy, target_energy) + 1)

            print(f"📊 PCA target energy={target_energy}, optimal k={k_opt}")

            # -----------------------------
            # 2. Fit final PCA with optimal k
            # -----------------------------
            pca = PCA(k=k_opt, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(train_normal_df)

            train_pca = pca_model.transform(train_normal_df)
            unlabeled_pca = pca_model.transform(unlabeled_df)

            # -----------------------------
            # 3. Reconstruction error UDF (squared error = better tails)
            # -----------------------------
            pc = pca_model.pc.toArray()

            @udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray())
                z = np.array(pca_vec.toArray())
                x_hat = np.dot(pc, z)
                return float(np.sum((x - x_hat) ** 2))

            train_pca = train_pca.withColumn("anomaly_score",
                reconstruction_error(col(feature_col), col("pca_features")))

            unlabeled_pca = unlabeled_pca.withColumn("anomaly_score",
                reconstruction_error(col(feature_col), col("pca_features")))

            # -----------------------------
            # 4. Contamination-based threshold (OCSVM equivalent)
            # -----------------------------
            expected_anomaly_rate = 0.01  # 🔧 try 0.01, 0.03, 0.05

            threshold = train_pca.approxQuantile("anomaly_score", [1 - expected_anomaly_rate], 0.01)[0]

            print(f"📏 PCA threshold (contamination={expected_anomaly_rate * 100:.1f}%): {threshold:.6f}")

            # -----------------------------
            # 5. Two-zone pseudo-labeling (reduce noise)
            # -----------------------------
            low_thr = train_pca.approxQuantile("anomaly_score", [0.95], 0.01)[0]

            print(f"📏 Two-zone thresholds: low={low_thr:.6f}, high={threshold:.6f}")

            pseudo_labels_df = unlabeled_pca.withColumn("pseudo_label",
                when(col("anomaly_score") >= threshold, 1).when(col("anomaly_score") <= low_thr, 0).otherwise(None))

            # Drop uncertain samples
            pseudo_labels_df = pseudo_labels_df.filter(col("pseudo_label").isNotNull())

            # -----------------------------
            # 6. Evaluate pseudo-labels (for analysis only)
            # -----------------------------
            unlabeled_eval_df = pseudo_labels_df.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")), on="Node_block_id",
                how="inner")

            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label").toPandas()
            y_true = pdf_unlabeled["true_label"]
            y_pred = pdf_unlabeled["pseudo_label"]

            print('Classification_report only for pseudo-label on unlabeled data (PCA robust)')
            print(classification_report(y_true, y_pred, digits=3))

            # -----------------------------
            # 7. Merge pseudo-labeled + normal
            # -----------------------------
            df_normal = train_normal_df.withColumn("Final_Label", when(col("Temp_label") == 0, 0))
            df_unlabeled = pseudo_labels_df.withColumnRenamed("pseudo_label", "Final_Label")

            df_final_train = df_normal.unionByName(df_unlabeled, allowMissingColumns=True)

            df_test = sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label"))

            df_final_train.printSchema()
            df_test.printSchema()

            # -----------------------------
            # 8. Full training label quality (diagnostic)
            # -----------------------------
            pdf_final = df_final_train.toPandas()
            y_train = pdf_final["Final_Label"].values
            y_train_truth = pdf_final["Label"].values

            print('Classification_report full training data (PCA robust novelty)')
            print(classification_report(y_train_truth, y_train, digits=3))
            '''



            from pyspark.ml.feature import PCA as SparkPCA


            # good ------
            print("\n🧠 Using PCA for novelty detection (semi-supervised) ...")

            # Train on normal logs only
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for PCA novelty detection.")

            feature_col = "features_vec_final"

            candidate_ks = [10, 20, 30, 50, 60, 70]
            target_variance = 0.999

            best_k = None

            for k in candidate_ks:
                print(f"[INFO] Testing PCA with k={k}")

                pca = SparkPCA(k=k, inputCol="features_vec_final", outputCol=f"pca_features_k{k}")

                pca_model = pca.fit(train_normal_df)

                # explainedVariance is a DenseVector of length k
                explained_variance = float(sum(pca_model.explainedVariance))

                print(f"[INFO] PCA k={k}, cumulative explained variance = {explained_variance:.6f}")

                # ✅ Take FIRST k that reaches target variance
                if explained_variance >= target_variance:
                    best_k = k
                    print(f"[SELECTED] First k reaching target variance: {best_k}")
                    break

            # Fallback if none reached target variance
            if best_k is None:
                best_k = candidate_ks[-1]
                print(f"[WARNING] Target variance not reached. Using max k = {best_k}")

            print(f"[RESULT] Selected PCA components (best_k): {best_k}")
            #exit()

            # -----------------------------
            # 1. Fit PCA on normal only
            # -----------------------------
            k_pca = best_k # 50  # 🔧 TUNE: try 20, 50, 100
            pca = PCA(k=k_pca, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(train_normal_df)

            # Transform
            train_pca = pca_model.transform(train_normal_df)
            unlabeled_pca = pca_model.transform(unlabeled_df)

            # -----------------------------
            # 2. Reconstruction error UDF
            # -----------------------------
            pc = pca_model.pc.toArray()

            @udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray())
                z = np.array(pca_vec.toArray())
                x_hat = np.dot(pc, z)
                return float(np.linalg.norm(x - x_hat))

            train_pca = train_pca.withColumn("anomaly_score",
                reconstruction_error(col(feature_col), col("pca_features")))

            unlabeled_pca = unlabeled_pca.withColumn("anomaly_score",
                reconstruction_error(col(feature_col), col("pca_features")))

            # -----------------------------
            # 3. Thresholds (from normal)
            # -----------------------------
            #threshold = train_pca.approxQuantile("anomaly_score", [0.90], 0.01)[0]
            #print(f"\n✅ PCA anomaly threshold (99% quantile of normal): {threshold:.6f}")
            # -----------------------------
            # 3. Thresholds (knee/elbow from normal)
            # -----------------------------
            # Collect normal reconstruction errors to driver
            scores = (train_pca.select("anomaly_score").toPandas()["anomaly_score"].astype(float).values)

            scores = np.sort(scores)
            x = np.arange(len(scores))
            knee = KneeLocator(x, scores, curve="convex", direction="increasing")
            knee_idx = knee.knee
            p_knee = (knee_idx + 1) / len(scores)  # approx percentile of knee
            p_use = max(p_knee - 0.01, 0.95)  # move 2% left, never below 90%
            threshold = float(np.quantile(scores, p_use))

            print(f"[INFO] p_knee≈{p_knee:.4f}, using p={p_use:.4f}, threshold={threshold:.6f}")

            #knee = KneeLocator(x, scores, curve="convex", direction="increasing")
#           #if knee.knee is None:
            #   # Fallback if knee not found (use a conservative high quantile)
            #   threshold = float(np.quantile(scores, 0.995))
            #    print("[WARNING] Knee not found. Fallback threshold = 99.5% quantile.")
            #else:
            #    knee_idx = knee.knee
            #    knee_thr = float(scores[knee_idx])
            #    alpha = 0.99  # try 0.9, 0.85, 0.8
            #    threshold = alpha * knee_thr

            #   print(f"[INFO] knee_thr={knee_thr:.6f}, relaxed threshold={threshold:.6f} (alpha={alpha})")

                #threshold = float(scores[knee.knee])
                #print(f"[INFO] Knee index = {knee.knee}")

            print(f"\n✅ PCA anomaly threshold (knee on normal): {threshold:.6f}")

            # -----------------------------
            # 4. Pseudo-labels (same style)
            # -----------------------------
            pseudo_labels_df = unlabeled_pca.withColumn("pseudo_label",
                when(col("anomaly_score") > threshold, 1).otherwise(0))

            # -----------------------------
            # 5. Evaluate pseudo-labels
            # -----------------------------
            unlabeled_eval_df = pseudo_labels_df.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")), on="Node_block_id",
                how="inner")

            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label").toPandas()

            y_true = pdf_unlabeled["true_label"]
            y_pred = pdf_unlabeled["pseudo_label"]

            print('Classification_report only for pseudo-label on unlabeled data (PCA)')
            print(classification_report(y_true, y_pred, digits=3))

            # -----------------------------
            # 6. Merge pseudo-labeled + normal
            # -----------------------------
            df_normal = train_normal_df.withColumn("Final_Label", when(col("Temp_label") == 0, 0))

            df_unlabeled = pseudo_labels_df.withColumnRenamed("pseudo_label", "Final_Label")

            df_final_train = df_normal.unionByName(df_unlabeled, allowMissingColumns=True)

            df_test = (sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label")))

            df_val = (sequences_df.filter(col("Temp_label") == 777).withColumn("Final_Label", col("Label")))

            df_final_train.printSchema()
            df_test.printSchema()
            #df_val.printSchema()

            # -----------------------------
            # 7. Full training label quality
            # -----------------------------
            pdf_final = df_final_train.toPandas()
            y_train = pdf_final["Final_Label"].values
            y_train_truth = pdf_final["Label"].values

            print('Classification_report full training data (PCA novelty)')
            print(classification_report(y_train_truth, y_train, digits=3))
            exit()


            '''
            print("\n☁️ Using KMeans Distance-based Novelty Detection ...")

            # Train on normal logs only
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for KMeans novelty detection.")

    
            feature_col = "features_vec_final"

            # -----------------------------
            # 1. Train KMeans on NORMAL data
            # -----------------------------
            k_values = [2,3, 5, 10,15, 20]
            best_model, best_score, best_k = None, -np.inf, None

            for k in k_values:
                print(f"🔍 Testing KMeans: k={k}")
                try:
                    kmeans = KMeans(featuresCol=feature_col, predictionCol="km_cluster", k=k, seed=42, maxIter=50)
                    model = kmeans.fit(train_normal_df)

                    # Use negative avg distance on normal as score (lower distance is better)
                    centers = model.clusterCenters()

                    def dist_to_center(cluster_id, features):
                        center = centers[int(cluster_id)]
                        return float(np.linalg.norm(np.array(features) - np.array(center)))

                    dist_udf = udf(dist_to_center, DoubleType())

                    preds_norm = model.transform(train_normal_df)
                    preds_norm = preds_norm.withColumn("dist", dist_udf(col("km_cluster"), col(feature_col)))

                    avg_dist = preds_norm.selectExpr("avg(dist) as avg_dist").collect()[0]["avg_dist"]

                    score = -avg_dist  # minimize distance
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_model = model

                except Exception as e:
                    print(f"⚠️ Failed for KMeans k={k}: {e}")
                    continue

            if best_model is None:
                raise RuntimeError("❌ No valid KMeans model found.")

            print(f"\n✅ Optimal KMeans k = {best_k}")
            print(f"   Best avg normal distance = {-best_score:.6f}")

            # -----------------------------
            # 2. Score UNLABELED using distance to centroid
            # -----------------------------
            centers = best_model.clusterCenters()

            def dist_to_center(cluster_id, features):
                center = centers[int(cluster_id)]
                return float(np.linalg.norm(np.array(features) - np.array(center)))

            dist_udf = udf(dist_to_center, DoubleType())

            preds = best_model.transform(unlabeled_df)
            preds = preds.withColumn("anomaly_score", dist_udf(col("km_cluster"), col(feature_col)))

            # -----------------------------
            # 3. Thresholding (percentile-based)
            # -----------------------------
            threshold = 0.09  #preds.approxQuantile("anomaly_score", [0.95], 0.01)[0]
            #print(f"🔥 KMeans anomaly threshold (95th pct): {threshold:.6f}")

            pseudo_labels_df = preds.withColumn("pseudo_label", when(col("anomaly_score") > threshold, 1).otherwise(0))

            # -----------------------------
            # 4. Evaluate pseudo-labels on unlabeled (if true labels exist)
            # -----------------------------
            unlabeled_eval_df = pseudo_labels_df.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")), on="Node_block_id",
                how="inner")

            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label").toPandas()
            y_true = pdf_unlabeled["true_label"]
            y_pred = pdf_unlabeled["pseudo_label"]

            print('Classification_report only for pseudo-labels for unlabeled data')
            print(classification_report(y_true, y_pred, digits=3))

            # -----------------------------
            # 5. Merge pseudo-labeled + normal logs
            # -----------------------------
            df_normal = train_normal_df.withColumn("Final_Label", when(col("Temp_label") == 0, 0))
            df_unlabeled = pseudo_labels_df.withColumnRenamed("pseudo_label", "Final_Label")

            df_final_train = df_normal.unionByName(df_unlabeled, allowMissingColumns=True)

            df_test = sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label"))

            df_val = sequences_df.filter(col("Temp_label") == 777) \
                .withColumn("Final_Label", col("Label"))

            df_final_train.printSchema()
            df_test.printSchema()
            df_val.printSchema()

            # -----------------------------
            # 6. Training label sanity check
            # -----------------------------
            pdf_final = df_final_train.toPandas()
            y_train = pdf_final["Final_Label"].values
            y_train_truth = pdf_final["Label"].values

            print('Classification_report full training data')
            print(classification_report(y_train_truth, y_train, digits=3))

            exit()
            '''

            '''
            print("\n☁️ Using Gaussian Mixture Model (semi-supervised) ...")
            # Train on normal logs only
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for GMM novelty detection.")

            feature_col = "features_vec_final"

            # GMM hyperparameter grid
            k_values = [2,3,5]  # [2,3,5,7,9,11]  # number of mixture components
            max_iter_values =  [5,10, 20, 50, 100,150, 200]
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
            '''



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

