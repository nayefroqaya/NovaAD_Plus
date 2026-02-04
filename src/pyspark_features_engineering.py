import colorama
import math
import numpy as np
import pandas as pd
import warnings
from kneed import KneeLocator
import numpy as np
from pyspark.sql.functions import abs as Fabs
from pyspark.sql.functions import col, lit, when, array_max, udf
from pyspark.sql.types import DoubleType
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit, array_max
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit, array_max
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

import numpy as np
import math
from kneed import KneeLocator
from sklearn.metrics import classification_report  # (optional, for debugging only)

import numpy as np
import math
from kneed import KneeLocator

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
import numpy as np

from pyspark.sql.functions import col, lit, when, array_max, udf
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.functions import vector_to_array

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
import numpy as np
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import BisectingKMeans
import numpy as np

import numpy as np
from pyspark.sql.functions import col, when, lit, array_max
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import classification_report

# Optional classifier stage (recommended to rescue TH_1G)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.sql.functions import col, when, udf, avg, stddev, lit
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import PCA as SparkPCA
import numpy as np
from kneed import KneeLocator

from pyspark.sql.functions import col, when, lit, array_max
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.functions import vector_to_array

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
    def novelty_detection_label_establishment(sequences_df: DataFrame,  spark: SparkSession, method: str = "gmm"
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
        val_df = sequences_df.filter(col("Temp_label") == 777)

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

        feature_col = "features_vec_final"

        #--------------------------------------------------------------------------------
        # Ensure feature column is vector type
        if method.lower() == "gmm":
            '''
            # good ----------------------------------------------------------------------------------------------------
            print("\n🧠 Using PCA for novelty detection (semi-supervised) ...")
            # Train on normal logs only
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)
            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for PCA novelty detection.")
            feature_col = "features_vec_final"
            # ------- Start the good idea ------------------------------------------------------------------------------
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
            # 3. Thresholds (knee/elbow from normal)
            # -----------------------------
            # Collect normal reconstruction errors to driver
            scores = (train_pca.select("anomaly_score").toPandas()["anomaly_score"].astype(float).values)
            scores = np.sort(scores)
            x = np.arange(len(scores))
            knee = KneeLocator(x, scores, curve="convex", direction="increasing")
            knee_idx = knee.knee
            p_knee = (knee_idx + 1) / len(scores)  # approx percentile of knee
            p_use = max(p_knee - 0.02, 0.95)  # move 2% left, never below 90%  # case1
            #p_use = min(0.999, max(p_knee + 0.01, 0.97))  # move RIGHT, conservative # case 2
            #p_use = min(0.999, max(p_knee + 0.03, 0.99))  # stronger conservative rule # case3
            #p_use = min(0.9999, max(p_knee + 0.03, 0.995))  # case 4
            threshold = float(np.quantile(scores, p_use))
            print(f"[INFO] p_knee≈{p_knee:.4f}, using p={p_use:.4f}, threshold={threshold:.6f}")
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
            #exit()
            # end good -------------------------------------------------------------------------------------------------
            '''
            '''
            #  good ---Idea2 -----------------------------------------------------------------------------------
            # -----------------------------
            # 0) Split data
            # -----------------------------
            print("\n🧠 Using PCA + GMM for novelty detection (semi-supervised) ...")

            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for novelty detection.")

            feature_col = "features_vec_final"

            # -----------------------------
            # 1) Choose PCA k on normal only (your "good idea")
            # -----------------------------
            candidate_ks = [10, 20, 30, 50, 60, 70]
            target_variance = 0.999

            best_k = None
            for k in candidate_ks:
                print(f"[INFO] Testing PCA with k={k}")
                pca_tmp = SparkPCA(k=k, inputCol=feature_col, outputCol=f"pca_features_k{k}")
                pca_tmp_model = pca_tmp.fit(train_normal_df)
                explained_variance = float(sum(pca_tmp_model.explainedVariance))
                print(f"[INFO] PCA k={k}, cumulative explained variance = {explained_variance:.6f}")
                if explained_variance >= target_variance:
                    best_k = k
                    print(f"[SELECTED] First k reaching target variance: {best_k}")
                    break

            if best_k is None:
                best_k = candidate_ks[-1]
                print(f"[WARNING] Target variance not reached. Using max k = {best_k}")

            print(f"[RESULT] Selected PCA components (best_k): {best_k}")

            # -----------------------------
            # 2) Fit PCA on normal only + transform normal/unlabeled
            # -----------------------------
            pca = SparkPCA(k=best_k, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(train_normal_df)

            train_pca = pca_model.transform(train_normal_df)
            unlabeled_pca = pca_model.transform(unlabeled_df)

            # -----------------------------
            # 3) PCA reconstruction error -> anomaly_score_pca
            # -----------------------------
            pc = pca_model.pc.toArray()

            @udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray(), dtype=float)
                z = np.array(pca_vec.toArray(), dtype=float)
                x_hat = np.dot(pc, z)
                return float(np.linalg.norm(x - x_hat))

            train_pca = train_pca.withColumn("anomaly_score_pca",
                                             reconstruction_error(col(feature_col), col("pca_features")))

            unlabeled_pca = unlabeled_pca.withColumn("anomaly_score_pca",
                                                     reconstruction_error(col(feature_col), col("pca_features")))

            # -----------------------------
            # 4) Threshold for PCA score using knee on normal
            # -----------------------------
            scores_pca = (train_pca.select("anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(float).values)
            scores_pca = np.sort(scores_pca)
            x = np.arange(len(scores_pca))

            knee = KneeLocator(x, scores_pca, curve="convex", direction="increasing")
            knee_idx = knee.knee
            p_knee = (knee_idx + 1) / len(scores_pca) if knee_idx is not None else 0.99

            p_use = max(p_knee - 0.02, 0.95)  # keep your default; change if you want
            thr_pca = float(np.quantile(scores_pca, p_use))

            print(f"[INFO] PCA p_knee≈{p_knee:.4f}, using p={p_use:.4f}, threshold={thr_pca:.6f}")

            unlabeled_pca = unlabeled_pca.withColumn("pseudo_label_pca",
                when(col("anomaly_score_pca") > thr_pca, 1).otherwise(0))

            # -----------------------------
            # 5) Fit GMM on NORMAL in PCA space + score unlabeled
            # -----------------------------
            gmm_k = 10  # 🔧 TUNE: 2, 3, 5, 8, 10
            gmm = GaussianMixture(k=gmm_k, featuresCol="pca_features", predictionCol="gmm_cluster",
                probabilityCol="gmm_prob")

            gmm_model = gmm.fit(train_pca.select("pca_features"))
            train_gmm = gmm_model.transform(train_pca)
            unlab_gmm = gmm_model.transform(unlabeled_pca)

            # membership confidence: max(probabilities); low -> anomaly
            train_gmm = train_gmm.withColumn("gmm_max_prob", array_max(vector_to_array(col("gmm_prob"))))
            unlab_gmm = unlab_gmm.withColumn("gmm_max_prob", array_max(vector_to_array(col("gmm_prob"))))

            train_gmm = train_gmm.withColumn("anomaly_score_gmm", lit(1.0) - col("gmm_max_prob"))
            unlab_gmm = unlab_gmm.withColumn("anomaly_score_gmm", lit(1.0) - col("gmm_max_prob"))

            # -----------------------------
            # 6) Threshold for GMM score using knee on normal (GMM)
            # -----------------------------
            scores_gmm = (train_gmm.select("anomaly_score_gmm").toPandas()["anomaly_score_gmm"].astype(float).values)
            scores_gmm = np.sort(scores_gmm)
            x2 = np.arange(len(scores_gmm))

            knee2 = KneeLocator(x2, scores_gmm, curve="convex", direction="increasing")
            knee2_idx = knee2.knee
            p2_knee = (knee2_idx + 1) / len(scores_gmm) if knee2_idx is not None else 0.99

            p2_use = max(p2_knee - 0.02, 0.95)
            thr_gmm = float(np.quantile(scores_gmm, p2_use))

            print(f"[INFO] GMM(k={gmm_k}) p_knee≈{p2_knee:.4f}, using p={p2_use:.4f}, threshold={thr_gmm:.6f}")

            unlab_gmm = unlab_gmm.withColumn("pseudo_label_gmm",
                when(col("anomaly_score_gmm") > thr_gmm, 1).otherwise(0))

            # -----------------------------
            # 7) Combine PCA + GMM pseudo-labels (choose ONE)
            # -----------------------------
            combine_rule = "AND"  # "AND" for higher precision, "OR" for higher recall

            if combine_rule.upper() == "AND":
                unlab_gmm = unlab_gmm.withColumn("pseudo_label_final",
                    when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1), 1).otherwise(0))
            else:
                unlab_gmm = unlab_gmm.withColumn("pseudo_label_final",
                    when((col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1), 1).otherwise(0))

            # -----------------------------
            # 8) Evaluate (PCA vs GMM vs Combined) on unlabeled (if you have true Label)
            # -----------------------------
            unlabeled_eval_df = unlab_gmm.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")), on="Node_block_id",
                how="inner")

            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label_pca", "pseudo_label_gmm",
                "pseudo_label_final").toPandas()

            print("\n=== Classification_report on unlabeled (PCA-only) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_pca"], digits=3))

            print("\n=== Classification_report on unlabeled (GMM-only) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_gmm"], digits=3))

            print("\n=== Classification_report on unlabeled (Combined) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_final"], digits=3))

            # -----------------------------
            # 9) Build final training set using COMBINED pseudo labels
            # -----------------------------
            df_normal = train_normal_df.withColumn("Final_Label", lit(0))
            df_unlabeled = unlab_gmm.withColumnRenamed("pseudo_label_final", "Final_Label")

            df_final_train = df_normal.unionByName(df_unlabeled, allowMissingColumns=True)

            df_test = sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label"))
            df_val = sequences_df.filter(col("Temp_label") == 777).withColumn("Final_Label", col("Label"))

            print("\n[INFO] Full training label quality (Combined pseudo labels):")
            pdf_final = df_final_train.select("Final_Label", "Label").toPandas()
            print(classification_report(pdf_final["Label"].values, pdf_final["Final_Label"].values, digits=3))
            exit()
            # end good ----------------idea 2------------------------------------------------------------------------
            '''
            '''
            # start  good ----------------idea 2--------case 2----------------------------------------------------------
            print("\n🧠 Using PCA + GMM for novelty detection (semi-supervised) ...")

            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for novelty detection.")

            feature_col = "features_vec_final"

            # -----------------------------
            # Helper: unsupervised model-quality score (Cohen's d * anomaly-rate penalty)
            # -----------------------------
            def unsup_score(df, score_col, label_col, r0=0.03, alpha=2.0):
                """
                Unsupervised quality score for a pseudo-labeling method:
                  - Separation in score space between pseudo-normal (0) and pseudo-anomaly (1) via Cohen's d
                  - Penalize anomaly rate far from expected r0 (log-ratio penalty)
                Returns float (higher = better).
                """
                stats = (df.groupBy(label_col).agg(F.count("*").alias("n"), F.avg(score_col).alias("mu"),
                                                   F.stddev_pop(score_col).alias("sigma"))).collect()

                s = {int(row[label_col]): row for row in stats if row[label_col] is not None}

                # if one side missing -> bad pseudo-labeler
                if 0 not in s or 1 not in s:
                    return float("-inf")

                mu0 = float(s[0]["mu"])
                mu1 = float(s[1]["mu"])
                sg0 = float(s[0]["sigma"] or 0.0)
                sg1 = float(s[1]["sigma"] or 0.0)
                n0 = int(s[0]["n"])
                n1 = int(s[1]["n"])
                r = n1 / max((n0 + n1), 1)

                pooled = math.sqrt((sg0 * sg0 + sg1 * sg1) / 2.0) + 1e-12
                d = (mu1 - mu0) / pooled  # separation

                penalty = math.exp(-alpha * abs(math.log((r + 1e-12) / r0)))
                return d * penalty

            # -----------------------------
            # 1) Choose PCA k on normal only
            # -----------------------------
            candidate_ks = [10, 20, 30, 50, 60, 70]
            target_variance = 0.999

            best_k = None
            for k in candidate_ks:
                print(f"[INFO] Testing PCA with k={k}")
                pca_tmp = SparkPCA(k=k, inputCol=feature_col, outputCol=f"pca_features_k{k}")
                pca_tmp_model = pca_tmp.fit(train_normal_df)
                explained_variance = float(sum(pca_tmp_model.explainedVariance))
                print(f"[INFO] PCA k={k}, cumulative explained variance = {explained_variance:.6f}")
                if explained_variance >= target_variance:
                    best_k = k
                    print(f"[SELECTED] First k reaching target variance: {best_k}")
                    break

            if best_k is None:
                best_k = candidate_ks[-1]
                print(f"[WARNING] Target variance not reached. Using max k = {best_k}")

            print(f"[RESULT] Selected PCA components (best_k): {best_k}")

            # -----------------------------
            # 2) Fit PCA on normal only + transform normal/unlabeled
            # -----------------------------
            pca = SparkPCA(k=best_k, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(train_normal_df)

            train_pca = pca_model.transform(train_normal_df)
            unlabeled_pca = pca_model.transform(unlabeled_df)

            # -----------------------------
            # 3) PCA reconstruction error -> anomaly_score_pca
            #     (kept as-is; note: Python UDF can be slow at scale)
            # -----------------------------
            pc = pca_model.pc.toArray()

            @udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray(), dtype=float)
                z = np.array(pca_vec.toArray(), dtype=float)
                x_hat = np.dot(pc, z)
                return float(np.linalg.norm(x - x_hat))

            train_pca = train_pca.withColumn("anomaly_score_pca",
                reconstruction_error(col(feature_col), col("pca_features")))

            unlabeled_pca = unlabeled_pca.withColumn("anomaly_score_pca",
                reconstruction_error(col(feature_col), col("pca_features")))

            # -----------------------------
            # 4) Threshold for PCA score using knee on normal (PCA)
            # -----------------------------
            scores_pca = (train_pca.select("anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(float).values)
            scores_pca = np.sort(scores_pca)
            x = np.arange(len(scores_pca))

            knee = KneeLocator(x, scores_pca, curve="convex", direction="increasing")
            knee_idx = knee.knee
            p_knee = (knee_idx + 1) / len(scores_pca) if knee_idx is not None else 0.99

            p_use = max(p_knee - 0.02, 0.95)
            thr_pca = float(np.quantile(scores_pca, p_use))

            print(f"[INFO] PCA p_knee≈{p_knee:.4f}, using p={p_use:.4f}, threshold={thr_pca:.6f}")

            unlabeled_pca = unlabeled_pca.withColumn("pseudo_label_pca",
                when(col("anomaly_score_pca") > thr_pca, 1).otherwise(0))

            # -----------------------------
            # 5) Fit GMM on NORMAL in PCA space + score unlabeled
            # -----------------------------
            gmm_k = 10  # 🔧 TUNE: 2, 3, 5, 8, 10
            gmm = GaussianMixture(k=gmm_k, featuresCol="pca_features", predictionCol="gmm_cluster",
                probabilityCol="gmm_prob")

            gmm_model = gmm.fit(train_pca.select("pca_features"))
            train_gmm = gmm_model.transform(train_pca)
            unlab_gmm = gmm_model.transform(unlabeled_pca)

            train_gmm = train_gmm.withColumn("gmm_max_prob", array_max(vector_to_array(col("gmm_prob"))))
            unlab_gmm = unlab_gmm.withColumn("gmm_max_prob", array_max(vector_to_array(col("gmm_prob"))))

            train_gmm = train_gmm.withColumn("anomaly_score_gmm", lit(1.0) - col("gmm_max_prob"))
            unlab_gmm = unlab_gmm.withColumn("anomaly_score_gmm", lit(1.0) - col("gmm_max_prob"))

            # -----------------------------
            # 6) Threshold for GMM score using knee on normal (GMM)
            # -----------------------------
            scores_gmm = (train_gmm.select("anomaly_score_gmm").toPandas()["anomaly_score_gmm"].astype(float).values)
            scores_gmm = np.sort(scores_gmm)
            x2 = np.arange(len(scores_gmm))

            knee2 = KneeLocator(x2, scores_gmm, curve="convex", direction="increasing")
            knee2_idx = knee2.knee
            p2_knee = (knee2_idx + 1) / len(scores_gmm) if knee2_idx is not None else 0.99

            p2_use = max(p2_knee - 0.02, 0.95)
            thr_gmm = float(np.quantile(scores_gmm, p2_use))

            print(f"[INFO] GMM(k={gmm_k}) p_knee≈{p2_knee:.4f}, using p={p2_use:.4f}, threshold={thr_gmm:.6f}")

            unlab_gmm = unlab_gmm.withColumn("pseudo_label_gmm",
                when(col("anomaly_score_gmm") > thr_gmm, 1).otherwise(0))

            # -----------------------------
            # 7) Combine PCA + GMM pseudo-labels (AND/OR)
            # -----------------------------
            combine_rule = "AND"  # "AND" for higher precision, "OR" for higher recall

            if combine_rule.upper() == "AND":
                unlab_gmm = unlab_gmm.withColumn("pseudo_label_final",
                    when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1), 1).otherwise(0))
            else:
                unlab_gmm = unlab_gmm.withColumn("pseudo_label_final",
                    when((col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1), 1).otherwise(0))

            # -----------------------------
            # 7.1) Build a Combined anomaly score (Spark-native)
            #   Normalize each score by its threshold (so they are comparable),
            #   then take max() as combined score.
            # -----------------------------
            unlab_gmm = (unlab_gmm.withColumn("score_pca_norm", col("anomaly_score_pca") / lit(thr_pca)).withColumn(
                "score_gmm_norm", col("anomaly_score_gmm") / lit(thr_gmm)).withColumn("anomaly_score_comb",
                                                                                      F.greatest(col("score_pca_norm"),
                                                                                                 col("score_gmm_norm"))))

            # -----------------------------
            # 7.2) Unsupervised quality scores to choose PCA vs GMM vs Combined
            # -----------------------------
            # We compute these on UNLABELED (no ground truth), using each method's own pseudo labels.
            score_method_pca = unsup_score(unlab_gmm, "anomaly_score_pca", "pseudo_label_pca")
            score_method_gmm = unsup_score(unlab_gmm, "anomaly_score_gmm", "pseudo_label_gmm")
            score_method_comb = unsup_score(unlab_gmm, "anomaly_score_comb", "pseudo_label_final")

            print("\n[UNSUPERVISED MODEL SELECTION SCORES] (higher is better)")
            print(f"  PCA-only     : {score_method_pca:.6f}")
            print(f"  GMM-only     : {score_method_gmm:.6f}")
            print(f"  Combined({combine_rule.upper()}) : {score_method_comb:.6f}")

            best_method = max([("pca", score_method_pca), ("gmm", score_method_gmm), ("combined", score_method_comb)],
                key=lambda x: x[1])[0]
            print(f"[SELECTED] Best method by unsupervised score: {best_method}")

            # -----------------------------
            # 7.3) Choose Final_Label based on selected method (unsupervised)
            # -----------------------------
            if best_method == "pca":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_pca"))
            elif best_method == "gmm":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_gmm"))
            else:
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_final"))

            # -----------------------------
            # 8) (Optional) Evaluate on unlabeled (ONLY if you have true Label, for debugging)
            # -----------------------------
            unlabeled_eval_df = unlab_gmm.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")), on="Node_block_id",
                how="inner")

            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label_pca", "pseudo_label_gmm",
                "pseudo_label_final", "Final_Label").toPandas()

            print("\n=== Classification_report on unlabeled (PCA-only) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_pca"], digits=3))

            print("\n=== Classification_report on unlabeled (GMM-only) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_gmm"], digits=3))

            print("\n=== Classification_report on unlabeled (Combined) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_final"], digits=3))

            print("\n=== Classification_report on unlabeled (SELECTED Final_Label) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["Final_Label"], digits=3))

            # -----------------------------
            # 9) Build final training set using SELECTED pseudo labels
            # -----------------------------
            df_normal = train_normal_df.withColumn("Final_Label", lit(0))
            df_unlabeled = unlab_gmm.select(*train_normal_df.columns, "Final_Label") if set(
                train_normal_df.columns).issubset(set(unlab_gmm.columns)) else unlab_gmm.withColumn("Final_Label",
                                                                                                    col("Final_Label"))

            df_final_train = df_normal.unionByName(unlab_gmm, allowMissingColumns=True)

            df_test = sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label"))
            df_val = sequences_df.filter(col("Temp_label") == 777).withColumn("Final_Label", col("Label"))

            print("\n[INFO] Full training label quality (Selected pseudo labels):")
            pdf_final = df_final_train.select("Final_Label", "Label").toPandas()
            print(classification_report(pdf_final["Label"].values, pdf_final["Final_Label"].values, digits=3))
            # end  good ----------------idea 2--------case 2----------------------------------------------------------
            '''
            # start  good ----------------idea 2--------case 3----------------------------------------------------------
            # -----------------------------
            # 0) Split data
            # -----------------------------
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            train_unlabeled_df = sequences_df.filter(col("Temp_label") == 999)
            df_test = sequences_df.filter(col("Temp_label") == 888)
            df_val = sequences_df.filter(col("Temp_label") == 777)


            if train_normal_df.count() == 0 or train_unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for novelty detection.")

            feature_col = "features_vec_final"
            id_col = "Node_block_id"

            # -----------------------------
            # Settings for unsupervised selection
            # -----------------------------
            EXPECTED_ANOM_RATE = 0.03  # r0: expected anomaly fraction (tune: 0.02, 0.05)
            RATE_ALPHA = 2.0  # penalty strength
            STABILITY_SEED1 = 21
            STABILITY_SEED2 = 99

            # -----------------------------
            # Helpers
            # -----------------------------
            def knee_threshold_from_scores(scores_np, min_quantile_floor=0.95, knee_margin=0.02,
                                           default_if_no_knee=0.99):
                """scores_np: 1D numpy array (unsorted ok). Returns (thr, p_knee, p_use)."""
                s = np.sort(scores_np.astype(float))
                x = np.arange(len(s))
                knee = KneeLocator(x, s, curve="convex", direction="increasing")
                knee_idx = knee.knee
                p_knee = (knee_idx + 1) / len(s) if knee_idx is not None else default_if_no_knee
                p_use = max(p_knee - knee_margin, min_quantile_floor)
                thr = float(np.quantile(s, p_use))
                return thr, float(p_knee), float(p_use)

            def split_df(df, frac=0.8, seed=42):
                a, b = df.randomSplit([frac, 1 - frac], seed=seed)
                return a, b

            def fpr_on_holdout(norm_holdout_df, score_col, thr):
                """Fraction of holdout normals flagged as anomalies by score>thr."""
                n = norm_holdout_df.count()
                if n == 0:
                    return 1.0
                fp = norm_holdout_df.filter(col(score_col) > lit(thr)).count()
                return fp / n

            def anomaly_rate(df, label_col):
                n = df.count()
                if n == 0:
                    return 0.0
                a = df.filter(col(label_col) == 1).count()
                return a / n

            def rate_penalty(r, r0=0.03, alpha=2.0):
                return math.exp(-alpha * abs(math.log((r + 1e-12) / r0)))

            def jaccard_anomaly_sets(df1, df2, id_col, label_col):
                a1 = df1.filter(col(label_col) == 1).select(id_col).distinct()
                a2 = df2.filter(col(label_col) == 1).select(id_col).distinct()
                inter = a1.join(a2, on=id_col, how="inner").count()
                union = a1.union(a2).distinct().count()
                return inter / union if union > 0 else 0.0

            # ============================================================
            # 1) Choose PCA k on NORMAL only
            # ============================================================
            candidate_ks = [10, 20, 30, 50, 60, 70]
            target_variance = 0.999

            best_k = None
            for k in candidate_ks:
                print(f"[INFO] Testing PCA with k={k}")
                pca_tmp = SparkPCA(k=k, inputCol=feature_col, outputCol=f"pca_features_k{k}")
                pca_tmp_model = pca_tmp.fit(train_normal_df)
                explained_variance = float(sum(pca_tmp_model.explainedVariance))
                print(f"[INFO] PCA k={k}, cumulative explained variance = {explained_variance:.6f}")
                if explained_variance >= target_variance:
                    best_k = k
                    print(f"[SELECTED] First k reaching target variance: {best_k}")
                    break

            if best_k is None:
                best_k = candidate_ks[-1]
                print(f"[WARNING] Target variance not reached. Using max k = {best_k}")

            print(f"[RESULT] Selected PCA components (best_k): {best_k}")

            # ============================================================
            # 2) Fit PCA on NORMAL + transform NORMAL/UNLABELED
            # ============================================================
            pca = SparkPCA(k=best_k, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(train_normal_df)

            train_pca_normal = pca_model.transform(train_normal_df)
            train_unlabeled_pca = pca_model.transform(train_unlabeled_df)
            test_pca = pca_model.transform(test_df)
            val_pca = pca_model.transform(val_df)

            # ============================================================
            # 3) PCA reconstruction error -> anomaly_score_pca
            #     (kept as-is; Python UDF can be slow at scale)
            # ============================================================
            pc = pca_model.pc.toArray()

            @udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray(), dtype=float)
                z = np.array(pca_vec.toArray(), dtype=float)
                x_hat = np.dot(pc, z)
                return float(np.linalg.norm(x - x_hat))

            train_pca_normal = train_pca_normal.withColumn("anomaly_score_pca",
                                             reconstruction_error(col(feature_col), col("pca_features")))
            train_unlabeled_pca = train_unlabeled_pca.withColumn("anomaly_score_pca",
                                                     reconstruction_error(col(feature_col), col("pca_features")))

            # ============================================================
            # 4) PCA threshold (knee on NORMAL)
            # ============================================================
            scores_pca_np = train_pca_normal.select("anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(float).values
            thr_pca, p_knee_pca, p_use_pca = knee_threshold_from_scores(scores_pca_np, min_quantile_floor=0.95,
                                                                        knee_margin=0.02)

            print(f"[INFO] PCA p_knee≈{p_knee_pca:.4f}, using p={p_use_pca:.4f}, threshold={thr_pca:.6f}")

            train_unlabeled_pca = train_unlabeled_pca.withColumn("pseudo_label_pca",
                                                     when(col("anomaly_score_pca") > lit(thr_pca), 1).otherwise(0))

            # ============================================================
            # 5) Fit GMM on NORMAL in PCA space + score UNLABELED
            # ============================================================
            gmm_k = 10  # 🔧 TUNE: 2, 3, 5, 8, 10
            gmm = GaussianMixture(k=gmm_k, featuresCol="pca_features", predictionCol="gmm_cluster",
                probabilityCol="gmm_prob")

            gmm_model = gmm.fit(train_pca_normal.select("pca_features"))
            train_gmm = gmm_model.transform(train_pca_normal)
            unlab_gmm = gmm_model.transform(train_unlabeled_pca)

            train_gmm = train_gmm.withColumn("gmm_max_prob", array_max(vector_to_array(col("gmm_prob"))))
            unlab_gmm = unlab_gmm.withColumn("gmm_max_prob", array_max(vector_to_array(col("gmm_prob"))))

            train_gmm = train_gmm.withColumn("anomaly_score_gmm", lit(1.0) - col("gmm_max_prob"))
            unlab_gmm = unlab_gmm.withColumn("anomaly_score_gmm", lit(1.0) - col("gmm_max_prob"))

            # ============================================================
            # 6) GMM threshold (knee on NORMAL)
            # ============================================================
            scores_gmm_np = train_gmm.select("anomaly_score_gmm").toPandas()["anomaly_score_gmm"].astype(float).values
            thr_gmm, p_knee_gmm, p_use_gmm = knee_threshold_from_scores(scores_gmm_np, min_quantile_floor=0.95,
                                                                        knee_margin=0.02)

            print(f"[INFO] GMM(k={gmm_k}) p_knee≈{p_knee_gmm:.4f}, using p={p_use_gmm:.4f}, threshold={thr_gmm:.6f}")

            unlab_gmm = unlab_gmm.withColumn("pseudo_label_gmm",
                                             when(col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0))

            # ============================================================
            # 7) Combine PCA + GMM pseudo-labels (AND/OR)
            # ============================================================
            combine_rule = "AND"  # "AND" for higher precision, "OR" for higher recall

            if combine_rule.upper() == "AND":
                unlab_gmm = unlab_gmm.withColumn("pseudo_label_final",
                    when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1), 1).otherwise(0))
            else:
                unlab_gmm = unlab_gmm.withColumn("pseudo_label_final",
                    when((col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1), 1).otherwise(0))

            # ============================================================
            # 7.1) NON-CIRCULAR unsupervised selection
            #      Score = (Stability / (Holdout Normal FPR + eps)) * ratePenalty
            # ============================================================
            # --- Holdout normal evaluation ---
            norm_fit_df, norm_holdout_df = split_df(train_normal_df, frac=0.8, seed=13)

            # Refit PCA on norm_fit_df
            pca_fit = SparkPCA(k=best_k, inputCol=feature_col, outputCol="pca_features")
            pca_fit_model = pca_fit.fit(norm_fit_df)
            pc_fit = pca_fit_model.pc.toArray()

            norm_fit_pca = pca_fit_model.transform(norm_fit_df)
            norm_hold_pca = pca_fit_model.transform(norm_holdout_df)

            @udf(DoubleType())
            def reconstruction_error_fit(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray(), dtype=float)
                z = np.array(pca_vec.toArray(), dtype=float)
                x_hat = np.dot(pc_fit, z)
                return float(np.linalg.norm(x - x_hat))

            norm_fit_pca = norm_fit_pca.withColumn("anomaly_score_pca",
                                                   reconstruction_error_fit(col(feature_col), col("pca_features")))
            norm_hold_pca = norm_hold_pca.withColumn("anomaly_score_pca",
                                                     reconstruction_error_fit(col(feature_col), col("pca_features")))

            # PCA threshold from norm_fit_pca
            scores_pca_fit_np = norm_fit_pca.select("anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(
                float).values
            thr_pca_fit, _, _ = knee_threshold_from_scores(scores_pca_fit_np, min_quantile_floor=0.95, knee_margin=0.02)
            fpr_pca = fpr_on_holdout(norm_hold_pca, "anomaly_score_pca", thr_pca_fit)

            # Refit GMM on norm_fit_pca
            gmm_fit = GaussianMixture(k=gmm_k, featuresCol="pca_features", predictionCol="gmm_cluster",
                                      probabilityCol="gmm_prob")
            gmm_fit_model = gmm_fit.fit(norm_fit_pca.select("pca_features"))

            norm_fit_gmm = gmm_fit_model.transform(norm_fit_pca).withColumn("gmm_max_prob",
                                                                            array_max(vector_to_array(col("gmm_prob"))))
            norm_hold_gmm = gmm_fit_model.transform(norm_hold_pca).withColumn("gmm_max_prob", array_max(
                vector_to_array(col("gmm_prob"))))

            norm_fit_gmm = norm_fit_gmm.withColumn("anomaly_score_gmm", lit(1.0) - col("gmm_max_prob"))
            norm_hold_gmm = norm_hold_gmm.withColumn("anomaly_score_gmm", lit(1.0) - col("gmm_max_prob"))

            scores_gmm_fit_np = norm_fit_gmm.select("anomaly_score_gmm").toPandas()["anomaly_score_gmm"].astype(
                float).values
            thr_gmm_fit, _, _ = knee_threshold_from_scores(scores_gmm_fit_np, min_quantile_floor=0.95, knee_margin=0.02)
            fpr_gmm = fpr_on_holdout(norm_hold_gmm, "anomaly_score_gmm", thr_gmm_fit)

            # Combined FPR on normals: apply AND/OR flags using holdout scores and fitted thresholds
            norm_hold_join = (norm_hold_gmm.select(id_col, "anomaly_score_gmm").join(
                norm_hold_pca.select(id_col, "anomaly_score_pca"), on=id_col, how="inner").withColumn("pca_flag", when(
                col("anomaly_score_pca") > lit(thr_pca_fit), 1).otherwise(0)).withColumn("gmm_flag", when(
                col("anomaly_score_gmm") > lit(thr_gmm_fit), 1).otherwise(0)))

            if combine_rule.upper() == "AND":
                norm_hold_join = norm_hold_join.withColumn("comb_flag",
                                                           when((col("pca_flag") == 1) & (col("gmm_flag") == 1),
                                                                1).otherwise(0))
            else:
                norm_hold_join = norm_hold_join.withColumn("comb_flag",
                                                           when((col("pca_flag") == 1) | (col("gmm_flag") == 1),
                                                                1).otherwise(0))

            fpr_comb = anomaly_rate(norm_hold_join, "comb_flag")

            print("\n[UNSUPERVISED HOLDOUT NORMAL FPR] (lower is better)")
            print(f"  PCA-only FPR     : {fpr_pca:.6f}")
            print(f"  GMM-only FPR     : {fpr_gmm:.6f}")
            print(f"  Combined({combine_rule.upper()}) FPR : {fpr_comb:.6f}")

            # --- Stability on unlabeled (cheap proxy) ---
            # (Optional but strong. Uses two random subsamples and compares anomaly set overlap.)
            u1 = unlab_gmm.sample(withReplacement=False, fraction=0.8, seed=STABILITY_SEED1)
            u2 = unlab_gmm.sample(withReplacement=False, fraction=0.8, seed=STABILITY_SEED2)

            stab_pca = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_pca")
            stab_gmm = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_gmm")
            stab_comb = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_final")

            print("\n[UNSUPERVISED STABILITY (Jaccard)] (higher is better)")
            print(f"  PCA-only     : {stab_pca:.4f}")
            print(f"  GMM-only     : {stab_gmm:.4f}")
            print(f"  Combined     : {stab_comb:.4f}")

            # --- Anomaly-rate penalty on unlabeled ---
            r_pca = anomaly_rate(unlab_gmm, "pseudo_label_pca")
            r_gmm = anomaly_rate(unlab_gmm, "pseudo_label_gmm")
            r_comb = anomaly_rate(unlab_gmm, "pseudo_label_final")

            pen_pca = rate_penalty(r_pca, r0=EXPECTED_ANOM_RATE, alpha=RATE_ALPHA)
            pen_gmm = rate_penalty(r_gmm, r0=EXPECTED_ANOM_RATE, alpha=RATE_ALPHA)
            pen_comb = rate_penalty(r_comb, r0=EXPECTED_ANOM_RATE, alpha=RATE_ALPHA)

            print("\n[UNLABELED ANOMALY RATES]")
            print(f"  PCA-only     : r={r_pca:.4f},  penalty={pen_pca:.4f}")
            print(f"  GMM-only     : r={r_gmm:.4f},  penalty={pen_gmm:.4f}")
            print(f"  Combined     : r={r_comb:.4f}, penalty={pen_comb:.4f}")

            # --- Final unsupervised selection score ---
            eps = 1e-6
            score_pca_unsup = (stab_pca / (fpr_pca + eps)) * pen_pca
            score_gmm_unsup = (stab_gmm / (fpr_gmm + eps)) * pen_gmm
            score_comb_unsup = (stab_comb / (fpr_comb + eps)) * pen_comb

            print("\n[FINAL UNSUPERVISED SELECTION SCORE] (higher is better)")
            print(f"  PCA-only     : {score_pca_unsup:.6f}")
            print(f"  GMM-only     : {score_gmm_unsup:.6f}")
            print(f"  Combined({combine_rule.upper()}) : {score_comb_unsup:.6f}")

            best_method = max([("pca", score_pca_unsup), ("gmm", score_gmm_unsup), ("combined", score_comb_unsup)],
                key=lambda x: x[1])[0]
            print(f"[SELECTED] Best method by NON-circular unsupervised score: {best_method}")

            # Assign Final_Label based on selected method
            if best_method == "pca":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_pca"))
            elif best_method == "gmm":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_gmm"))
            else:
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_final"))

            # ============================================================
            # 8) (Optional) Evaluate on unlabeled if you have true Label (DEBUG ONLY)
            # ============================================================
            train_unlabeled_eval_df = unlab_gmm.join(sequences_df.select(col(id_col), col("Label").alias("true_label")),
                on=id_col, how="inner")

            pdf_unlabeled = train_unlabeled_eval_df.select("true_label", "pseudo_label_pca", "pseudo_label_gmm",
                "pseudo_label_final", "Final_Label").toPandas()

            print("\n=== Classification_report on unlabeled (PCA-only) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_pca"], digits=3))

            print("\n=== Classification_report on unlabeled (GMM-only) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_gmm"], digits=3))

            print("\n=== Classification_report on unlabeled (Combined) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_final"], digits=3))

            print("\n=== Classification_report on unlabeled (SELECTED Final_Label) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["Final_Label"], digits=3))
            exit()

            # ============================================================
            # 9) Build final training set using SELECTED pseudo labels
            # ============================================================
            # Normal part MUST come from train_pca_normal (has pca_features)
            train_df_normal = train_pca_normal.withColumn("Final_Label", lit(0))
            # Unlabeled already has pca_features and Final_Label
            train_df_unlabeled = unlab_gmm
            keep_cols = [id_col, "pca_features", "Final_Label"]
            df_final_train_cls = train_df_normal.select(*keep_cols).unionByName(train_df_unlabeled.select(*keep_cols),
                allowMissingColumns=False)
            df_test_cls = (test_pca.withColumn("Final_Label", col("Label").cast("int")).select(id_col, "pca_features",
                                                                                               "Final_Label"))

            df_val_cls = (val_pca.withColumn("Final_Label", col("Label").cast("int")).select(id_col, "pca_features",
                                                                                             "Final_Label"))

            print('df_normal------')
            train_pca_normal.printSchema()
            print('df_unlabeled from train------')
            train_df_unlabeled.printSchema()
            print('full train------')
            df_final_train_cls.printSchema()
            print('full test------')
            df_test_cls.printSchema()
            print('full val------')
            df_val_cls.printSchema()
            exit()



            print("\n[INFO] Full training label quality (Selected pseudo labels) [DEBUG ONLY]:")
            pdf_final = df_final_train.select("Final_Label", "Label").toPandas()
            print(classification_report(pdf_final["Label"].values, pdf_final["Final_Label"].values, digits=3))


            print(f"\n✅ Novelty detection (GMM) completed successfully.")
            #exit()

            return df_final_train, df_test ,df_val  #, X_train, y_train, X_test, y_test_truth, X_val, y_val_truth

