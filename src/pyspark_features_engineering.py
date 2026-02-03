import colorama
import math
import numpy as np
import pandas as pd
import warnings
from kneed import KneeLocator
import numpy as np
from pyspark.sql.functions import abs as Fabs

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
            # start good ----------------idea 3-----------------------------------------------------------------------

            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            unlabeled_df = sequences_df.filter(col("Temp_label") == 999)

            if train_normal_df.count() == 0 or unlabeled_df.count() == 0:
                raise ValueError("❌ Not enough data for novelty detection.")

            feature_col = "features_vec_final"

            # -----------------------------
            # 1) Choose PCA k on normal only (your good idea)
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
            # 3) PCA reconstruction error -> score_pca
            # -----------------------------
            pc = pca_model.pc.toArray()

            @udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray(), dtype=float)
                z = np.array(pca_vec.toArray(), dtype=float)
                x_hat = np.dot(pc, z)
                return float(np.linalg.norm(x - x_hat))

            train_pca = train_pca.withColumn("score_pca", reconstruction_error(col(feature_col), col("pca_features")))
            unlabeled_pca = unlabeled_pca.withColumn("score_pca",
                                                     reconstruction_error(col(feature_col), col("pca_features")))

            # -----------------------------
            # 4) Fit GMM on NORMAL in PCA space + score unlabeled
            # -----------------------------
            gmm_k = 10  # fixed for paper (e.g., 5). (You can pick 5 or 10 and keep it fixed.)
            gmm = GaussianMixture(k=gmm_k, featuresCol="pca_features", predictionCol="gmm_cluster",
                probabilityCol="gmm_prob")

            gmm_model = gmm.fit(train_pca.select("pca_features"))

            train_gmm = gmm_model.transform(train_pca)
            unlab_gmm = gmm_model.transform(unlabeled_pca)

            # membership confidence: max(probabilities); low -> anomaly
            train_gmm = train_gmm.withColumn("gmm_max_prob", array_max(vector_to_array(col("gmm_prob"))))
            unlab_gmm = unlab_gmm.withColumn("gmm_max_prob", array_max(vector_to_array(col("gmm_prob"))))

            train_gmm = train_gmm.withColumn("score_gmm", lit(1.0) - col("gmm_max_prob"))
            unlab_gmm = unlab_gmm.withColumn("score_gmm", lit(1.0) - col("gmm_max_prob"))

            # -----------------------------
            # 5) Robust normalization using NORMAL stats (median + MAD)
            # -----------------------------
            pdf_norm = train_gmm.select("score_pca", "score_gmm").toPandas()

            def median_mad(arr):
                arr = np.asarray(arr, dtype=float)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med))) + 1e-12
                scale = 1.4826 * mad
                return med, scale

            med_pca, sc_pca = median_mad(pdf_norm["score_pca"].values)
            med_gmm, sc_gmm = median_mad(pdf_norm["score_gmm"].values)

            @udf(DoubleType())
            def z_pca(v):
                return float((float(v) - med_pca) / sc_pca)

            @udf(DoubleType())
            def z_gmm(v):
                return float((float(v) - med_gmm) / sc_gmm)

            train_gmm = train_gmm.withColumn("z_pca", z_pca(col("score_pca"))).withColumn("z_gmm",
                                                                                          z_gmm(col("score_gmm")))
            unlab_gmm = unlab_gmm.withColumn("z_pca", z_pca(col("score_pca"))).withColumn("z_gmm",
                                                                                          z_gmm(col("score_gmm")))

            # -----------------------------
            # 6) Fuse into one score (fixed weight for paper)
            # -----------------------------
            w = 0.5
            train_gmm = train_gmm.withColumn("fused_score", lit(w) * col("z_pca") + lit(1.0 - w) * col("z_gmm"))
            unlab_gmm = unlab_gmm.withColumn("fused_score", lit(w) * col("z_pca") + lit(1.0 - w) * col("z_gmm"))

            # -----------------------------
            # 7) ONE unsupervised threshold for all datasets: rate-cap on unlabeled
            # -----------------------------
            max_rate = 0.20  # fixed for paper (e.g., 10% maximum anomalies in unlabeled)
            thr = unlab_gmm.approxQuantile("fused_score", [1.0 - max_rate], 0.001)[0]

            print(f"[INFO] fused_score threshold by rate-cap (max_rate={max_rate:.1%}): {thr:.6f}")

            unlab_gmm = unlab_gmm.withColumn("pseudo_label_final", when(col("fused_score") > thr, 1).otherwise(0))

            # sanity check
            flagged = unlab_gmm.filter(col("pseudo_label_final") == 1).count()
            total = unlab_gmm.count()
            print(f"[INFO] flagged anomalies in unlabeled: {flagged}/{total} = {flagged / total:.3%}")

            # -----------------------------
            # 8) Evaluate pseudo-labels on unlabeled (if true Label exists)
            # -----------------------------
            unlabeled_eval_df = unlab_gmm.join(
                sequences_df.select(col("Node_block_id"), col("Label").alias("true_label")), on="Node_block_id",
                how="inner")

            pdf_unlabeled = unlabeled_eval_df.select("true_label", "pseudo_label_final").toPandas()
            print("\n=== Classification_report on unlabeled (FUSED novelty) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_final"], digits=3))

            # -----------------------------
            # 9) Build final training set (normal + pseudo-labeled)
            # -----------------------------
            df_normal = train_gmm.withColumn("Final_Label", lit(0)).withColumn("weight", lit(1.0))

            # confidence weighting (optional but helps TH_1G)
            # weight = 1 + |fused_score - threshold| ; anomalies get a bit more weight

            unlab_gmm = unlab_gmm.withColumn("conf", Fabs(col("fused_score") - lit(thr)))
            unlab_gmm = unlab_gmm.withColumn("weight", (lit(1.0) + col("conf")) * when(col("pseudo_label_final") == 1,
                                                                                       lit(2.0)).otherwise(lit(1.0)))

            df_unlabeled = unlab_gmm.withColumnRenamed("pseudo_label_final", "Final_Label")

            df_final_train = df_normal.unionByName(df_unlabeled, allowMissingColumns=True)

            df_test = sequences_df.filter(col("Temp_label") == 888).withColumn("Final_Label", col("Label"))
            df_val = sequences_df.filter(col("Temp_label") == 777).withColumn("Final_Label", col("Label"))

            # -----------------------------
            # 10) OPTIONAL (Recommended): Train ONE classifier for ALL datasets (GBT)
            #     Uses original features + scores to improve hard datasets (TH_1G)
            # -----------------------------
            USE_CLASSIFIER = True

            if USE_CLASSIFIER:
                # Need to carry score columns into test/val as well
                test_pca = pca_model.transform(df_test).withColumn("score_pca", reconstruction_error(col(feature_col),
                                                                                                     col("pca_features")))
                test_gmm = gmm_model.transform(test_pca).withColumn("gmm_max_prob",
                                                                    array_max(vector_to_array(col("gmm_prob"))))
                test_gmm = test_gmm.withColumn("score_gmm", lit(1.0) - col("gmm_max_prob"))
                test_gmm = test_gmm.withColumn("z_pca", z_pca(col("score_pca"))).withColumn("z_gmm",
                                                                                            z_gmm(col("score_gmm")))
                test_gmm = test_gmm.withColumn("fused_score", lit(w) * col("z_pca") + lit(1.0 - w) * col("z_gmm"))

                assembler = VectorAssembler(inputCols=[feature_col, "score_pca", "score_gmm", "fused_score"],
                    outputCol="features_all")

                train_ready = assembler.transform(df_final_train)
                test_ready = assembler.transform(test_gmm)

                gbt = GBTClassifier(labelCol="Final_Label", featuresCol="features_all", weightCol="weight", maxIter=80,
                    maxDepth=5, seed=42)

                model = gbt.fit(train_ready)
                pred_test = model.transform(test_ready)

                pdf_test = pred_test.select(col("Final_Label").alias("y_true"),
                                            col("prediction").alias("y_pred")).toPandas()
                print("\n=== Test Classification_report (GBT after FUSED novelty) ===")
                print(classification_report(pdf_test["y_true"], pdf_test["y_pred"], digits=3))

            # -----------------------------
            # 11) Training label quality report (pseudo labels vs true labels in train+unlabeled)
            # -----------------------------
            pdf_final = df_final_train.select("Final_Label", "Label").toPandas()
            print("\n[INFO] Full training label quality (FUSED pseudo labels):")
            print(classification_report(pdf_final["Label"].values, pdf_final["Final_Label"].values, digits=3))

            exit()











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

