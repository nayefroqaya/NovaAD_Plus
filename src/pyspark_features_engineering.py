import warnings

import colorama
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import pandas as pd
from kneed import KneeLocator
from kneed import KneeLocator
from kneed import KneeLocator
from kneed import KneeLocator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import PCA
from pyspark.ml.feature import PCA
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import RobustScaler
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
# Optional classifier stage (recommended to rescue TH_1G)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import VectorSizeHint, VectorAssembler
from pyspark.ml.functions import array_to_vector
from pyspark.ml.functions import vector_to_array

from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import functions as F
from pyspark.sql import functions as F
from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import abs as Fabs
from pyspark.sql.functions import array, col, concat
from pyspark.sql.functions import array_max, col, when
from pyspark.sql.functions import col, avg, when
from pyspark.sql.functions import col, lit, when, array_max, udf
from pyspark.sql.functions import col, lit, when, array_max, udf
from pyspark.sql.functions import col, size, max as spark_max
from pyspark.sql.functions import col, sum as spark_sum
from pyspark.sql.functions import col, udf
from pyspark.sql.functions import col, when
from pyspark.sql.functions import col, when, array
from pyspark.sql.functions import col, when, array_max
from pyspark.sql.functions import col, when, avg, udf
from pyspark.sql.functions import col, when, count, trim
from pyspark.sql.functions import col, when, lit, array_max
from pyspark.sql.functions import col, when, lit, array_max
from pyspark.sql.functions import col, when, lit, array_max
from pyspark.sql.functions import col, when, lit, array_max
from pyspark.sql.functions import col, when, udf
from pyspark.sql.functions import col, when, udf
from pyspark.sql.functions import col, when, udf
from pyspark.sql.functions import col, when, udf, avg, stddev, lit
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import udf
from pyspark.sql.functions import udf
from pyspark.sql.functions import udf
from pyspark.sql.functions import udf
from pyspark.sql.functions import udf
from pyspark.sql.functions import udf
from pyspark.sql.functions import udf
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import *
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import FloatType
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, DoubleType
from scipy.stats import chi2
from sklearn.metrics import classification_report  # (optional, for debugging only)
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
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
                                              count(when(col("sentiment_label").isNotNull(), True)).alias(
                                                  "not_null_count"))

        print("Columns:", final_train_with_test_with_val.columns)
        final_train_with_test_with_val.select("sentiment_label").printSchema()

        final_train_with_test_with_val = final_train_with_test_with_val.withColumn("sentiment_label", when(
            col("sentiment_label").isNull() | (trim(col("sentiment_label")) == "") | (
                col("sentiment_label").isin("None", "NaN", "null")), "negative").otherwise(col("sentiment_label")))

        # Normalize and clean first (optional but safer)
        final_train_with_test_with_val = final_train_with_test_with_val.withColumn("sentiment_label",
                                                                                   trim(col("sentiment_label")))

        # Then map to numeric
        final_train_with_test_with_val = final_train_with_test_with_val.withColumn("sentiment_label_indexed", when(
            col("sentiment_label") == "positive", 1).otherwise(0))

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
        summed_df_normal_labelled_train = (
        log_normal_labelled.groupby("Node_block_id").agg(avg_vector_udf(F.collect_list("features")).alias("features")))

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
        summed_df_combine_unlabelled_test = (
        log_test_unlabelled.groupby("Node_block_id").agg(avg_vector_udf(F.collect_list("features")).alias("features")))

        sequence_labels_combine_unlabelled_test = (log_test_unlabelled.groupby("Node_block_id").agg(
            F.when(F.sum(F.when(F.col("Label") != "Normal", 1).otherwise(0)) > 0, "anomaly").otherwise("normal").alias(
                "Label")))

        summed_df_combine_unlabelled_test = (
            summed_df_combine_unlabelled_test.join(sequence_labels_combine_unlabelled_test, "Node_block_id",
                                                   "inner").withColumn("Temp_label", F.lit(888)))

        summed_df_test = summed_df_combine_unlabelled_test

        # ------------------ (4) validate dataset (Temp_label = 777) -----------------------
        summed_df_combine_labelled_val = (
        log_val_labelled.groupby("Node_block_id").agg(avg_vector_udf(F.collect_list("features")).alias("features")))

        sequence_labels_combine_labelled_val = (log_val_labelled.groupby("Node_block_id").agg(
            F.when(F.sum(F.when(F.col("Label") != "Normal", 1).otherwise(0)) > 0, "anomaly").otherwise("normal").alias(
                "Label")))

        summed_df_combine_labelled_val = (
            summed_df_combine_labelled_val.join(sequence_labels_combine_labelled_val, "Node_block_id",
                                                "inner").withColumn("Temp_label", F.lit(777)))

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
    def novelty_detection_label_establishment(sequences_df: DataFrame, spark: SparkSession, method: str = "gmm"
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

        print('info labe  ..........................................')
        train_normal_df.groupBy("Label").count().show()
        unlabeled_train_df.groupBy("Label").count().show()
        test_df.groupBy("Label").count().show()
        #        exit()
        if train_normal_df.count() == 0:
            raise ValueError("❌ No normal logs (Temp_label=0) found for training.")
        if unlabeled_train_df.count() == 0:
            raise ValueError("❌ No unlabeled logs (Temp_label=999) found for novelty detection.")

        # --------------------------------------------------------------------------------
        # Ensure feature column is vector type
        if method.lower() == "gmm":

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
            # Settings
            # -----------------------------
            TARGET_FPR = 0.01  # main knob: 0.005 / 0.001 for fewer false alarms; 0.02 for more recall
            eps = 1e-9

            print("\n🧠 Using PCA + GMM for novelty detection (stable version) ...")

            # ============================================================
            # 0.1) Deterministic key + deterministic normal split (80/20)
            # ============================================================
            train_normal_df = train_normal_df.withColumn("hid", F.xxhash64(col(id_col)))
            train_unlabeled_df = train_unlabeled_df.withColumn("hid", F.xxhash64(col(id_col)))
            df_test = df_test.withColumn("hid", F.xxhash64(col(id_col)))
            df_val = df_val.withColumn("hid", F.xxhash64(col(id_col)))

            norm_fit_df = train_normal_df.filter((col("hid") % lit(100)) < lit(80))
            norm_holdout_df = train_normal_df.filter((col("hid") % lit(100)) >= lit(80))

            # -----------------------------
            # Helpers
            # -----------------------------
            def approx_quantile(df, c, q, rel=1e-3):
                return float(df.approxQuantile(c, [q], rel)[0])

            def threshold_from_norm_fit(norm_fit_scored_df, score_col, target_fpr=0.01):
                return approx_quantile(norm_fit_scored_df, score_col, 1.0 - target_fpr)

            def fpr_on_holdout(norm_holdout_scored_df, score_col, thr):
                n = norm_holdout_scored_df.count()
                if n == 0:
                    return 1.0
                fp = norm_holdout_scored_df.filter(col(score_col) > lit(thr)).count()
                return fp / n

            def anomaly_rate(df, label_col):
                n = df.count()
                if n == 0:
                    return 0.0
                a = df.filter(col(label_col) == 1).count()
                return a / n

            def jaccard_anomaly_sets(df1, df2, id_col, label_col):
                a1 = df1.filter(col(label_col) == 1).select(id_col).distinct()
                a2 = df2.filter(col(label_col) == 1).select(id_col).distinct()
                inter = a1.join(a2, on=id_col, how="inner").count()
                union = a1.union(a2).distinct().count()
                return inter / union if union > 0 else 0.0

            def separation_z(norm_scores, unlab_scores):
                mu_n = float(np.mean(norm_scores))
                sd_n = float(np.std(norm_scores)) + eps
                mu_u = float(np.mean(unlab_scores))
                return max(0.0, (mu_u - mu_n) / sd_n)

            # ============================================================
            # 1) Choose PCA k on NORMAL-fit only
            # ============================================================
           # candidate_ks = [10, 20, 40, 50, 60, 70]
            candidate_ks = [10, 20, 40, 50]
            #candidate_ks = [ 50, 60, 70, 80,90]
            target_variance = 0.999

            best_k = None
            for k in candidate_ks:
                print(f"[INFO] Testing PCA with k={k}")
                pca_tmp = SparkPCA(k=k, inputCol=feature_col, outputCol=f"pca_features_k{k}")
                pca_tmp_model = pca_tmp.fit(norm_fit_df)
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
            # 2) Fit PCA on NORMAL-fit + transform all
            # ============================================================
            pca = SparkPCA(k=best_k, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(norm_fit_df)

            train_pca_normal_fit = pca_model.transform(norm_fit_df)
            train_pca_normal_hold = pca_model.transform(norm_holdout_df)

            train_unlabeled_pca = pca_model.transform(train_unlabeled_df)
            test_pca = pca_model.transform(df_test)
            val_pca = pca_model.transform(df_val)

            # ============================================================
            # 3) PCA reconstruction error (handles pc orientation)
            # ============================================================
            pc = pca_model.pc.toArray()

            # feature dimension d
            d = len(train_pca_normal_fit.select(feature_col).head()[0])
            use_pc_dk = (pc.shape[0] == d)  # if pc is (d,k) then pc @ z else z @ pc

            pc_b = spark.sparkContext.broadcast(pc)
            use_pc_dk_b = spark.sparkContext.broadcast(use_pc_dk)

            @udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray(), dtype=float)
                z = np.array(pca_vec.toArray(), dtype=float)
                pc_local = pc_b.value
                if use_pc_dk_b.value:
                    x_hat = pc_local @ z
                else:
                    x_hat = z @ pc_local
                return float(np.linalg.norm(x - x_hat))

            train_pca_normal_fit = train_pca_normal_fit.withColumn("anomaly_score_pca",
                reconstruction_error(col(feature_col), col("pca_features")))
            train_pca_normal_hold = train_pca_normal_hold.withColumn("anomaly_score_pca",
                reconstruction_error(col(feature_col), col("pca_features")))
            train_unlabeled_pca = train_unlabeled_pca.withColumn("anomaly_score_pca",
                reconstruction_error(col(feature_col), col("pca_features")))

            # ============================================================
            # 4) PCA threshold (FPR-controlled)
            # ============================================================
            thr_pca = threshold_from_norm_fit(train_pca_normal_fit, "anomaly_score_pca", target_fpr=TARGET_FPR)
            fpr_pca = fpr_on_holdout(train_pca_normal_hold, "anomaly_score_pca", thr_pca)
            print(f"[PCA] thr={thr_pca:.6f}, holdout FPR={fpr_pca:.6f}")

            train_unlabeled_pca = train_unlabeled_pca.withColumn("pseudo_label_pca",
                when(col("anomaly_score_pca") > lit(thr_pca), 1).otherwise(0))

            # ============================================================
            # 5) Fit GMM on NORMAL-fit PCA space (choose k by BIC), score by NLL
            # ============================================================
            gmm_ks = [2, 4, 6, 8, 10]
            SEED = 123

            def bic_from_model(model, df_features, k, d_pca):
                try:
                    ll = float(model.summary.logLikelihood)
                except Exception:
                    return float("inf")
                n = df_features.count()
                p = (k - 1) + (k * d_pca) + (k * (d_pca * (d_pca + 1) // 2))
                return -2.0 * ll + p * np.log(max(n, 1))

            feat_df = train_pca_normal_fit.select("pca_features")
            d_pca = best_k

            best_gmm_model = None
            best_bic = float("inf")
            best_gmm_k = None

            for k in gmm_ks:
                gmm = GaussianMixture(k=k, seed=SEED, featuresCol="pca_features", predictionCol="gmm_cluster",
                    probabilityCol="gmm_prob")
                model = gmm.fit(feat_df)
                bic = bic_from_model(model, feat_df, k, d_pca)
                print(f"[GMM] k={k}, BIC={bic}")
                if bic < best_bic:
                    best_bic = bic
                    best_gmm_model = model
                    best_gmm_k = k

            if best_gmm_model is None:
                raise RuntimeError("❌ GMM selection failed (logLikelihood unavailable).")

            print(f"[GMM] Selected k={best_gmm_k} by BIC.")

            # ---- Robust extraction of means/covs (FIX for your error) ----
            def vec_to_np(v):
                return np.array(v.toArray(), dtype=float)

            def cov_to_2d(cov, d_dim):
                if hasattr(cov, "toArray"):
                    A = np.array(cov.toArray(), dtype=float)
                    if A.ndim == 2:
                        return A
                    if A.ndim == 1 and A.size == d_dim * d_dim:
                        return A.reshape(d_dim, d_dim)
                    raise ValueError(f"Unexpected cov shape from toArray: {A.shape}")
                A = np.array(cov, dtype=float)
                if A.ndim == 2 and A.shape == (d_dim, d_dim):
                    return A
                if A.ndim == 1 and A.size == d_dim * d_dim:
                    return A.reshape(d_dim, d_dim)
                raise ValueError(f"Could not convert covariance to (d,d). Got shape {A.shape}")

            weights = np.array(best_gmm_model.weights, dtype=float)
            gaussians = best_gmm_model.gaussians

            means = np.stack([vec_to_np(g.mean) for g in gaussians], axis=0)  # (k,d)
            covs = np.stack([cov_to_2d(g.cov, d_pca) for g in gaussians], axis=0)  # (k,d,d)

            # jitter to avoid singular matrices
            JITTER = 1e-6
            covs = covs + np.eye(d_pca)[None, :, :] * JITTER

            inv_covs = np.linalg.inv(covs)
            sign, logdets = np.linalg.slogdet(covs)
            if np.any(sign <= 0):
                covs = covs + np.eye(d_pca)[None, :, :] * (JITTER * 100)
                inv_covs = np.linalg.inv(covs)
                sign, logdets = np.linalg.slogdet(covs)

            const = d_pca * np.log(2.0 * np.pi)

            bc_params = spark.sparkContext.broadcast(
                {"weights": weights, "means": means, "inv_covs": inv_covs, "logdets": logdets, "const": const})

            @udf(DoubleType())
            def gmm_nll(pca_vec):
                z = np.array(pca_vec.toArray(), dtype=float)
                P = bc_params.value
                w = P["weights"];
                m = P["means"];
                ic = P["inv_covs"];
                ld = P["logdets"]
                cst = float(P["const"])

                logps = []
                for i in range(len(w)):
                    diff = z - m[i]
                    quad = float(diff.T @ ic[i] @ diff)
                    logN = -0.5 * (quad + float(ld[i]) + cst)
                    logps.append(np.log(max(float(w[i]), 1e-300)) + logN)

                a = float(np.max(logps))
                logp = a + float(np.log(np.sum(np.exp(np.array(logps) - a))))
                return float(-logp)  # higher => more anomalous

            # Score normals and unlabeled with GMM-NLL
            train_gmm_fit = train_pca_normal_fit.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            train_gmm_hold = train_pca_normal_hold.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            unlab_gmm = train_unlabeled_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))

            # ============================================================
            # 6) GMM threshold (FPR-controlled)
            # ============================================================
            thr_gmm = threshold_from_norm_fit(train_gmm_fit, "anomaly_score_gmm", target_fpr=TARGET_FPR)
            fpr_gmm = fpr_on_holdout(train_gmm_hold, "anomaly_score_gmm", thr_gmm)
            print(f"[GMM-NLL] thr={thr_gmm:.6f}, holdout FPR={fpr_gmm:.6f}")

            unlab_gmm = unlab_gmm.withColumn("pseudo_label_gmm",
                when(col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0))

            # ============================================================
            # 7) Create BOTH combined rules (AND + OR)
            # ============================================================
            unlab_gmm = unlab_gmm.withColumn("pseudo_label_and",
                when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1), 1).otherwise(0))
            unlab_gmm = unlab_gmm.withColumn("pseudo_label_or",
                when((col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1), 1).otherwise(0))

            # ============================================================
            # 7.1) Holdout-normal FPR computation (PCA/GMM/AND/OR)
            # ============================================================
            norm_hold_join = (train_gmm_hold.select(id_col, "anomaly_score_gmm").join(
                train_pca_normal_hold.select(id_col, "anomaly_score_pca"), on=id_col, how="inner").withColumn(
                "pca_flag", when(col("anomaly_score_pca") > lit(thr_pca), 1).otherwise(0)).withColumn("gmm_flag", when(
                col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0)).withColumn("and_flag", when(
                (col("pca_flag") == 1) & (col("gmm_flag") == 1), 1).otherwise(0)).withColumn("or_flag", when(
                (col("pca_flag") == 1) | (col("gmm_flag") == 1), 1).otherwise(0)))

            fpr_and = anomaly_rate(norm_hold_join, "and_flag")
            fpr_or = anomaly_rate(norm_hold_join, "or_flag")

            print("\n[HOLDOUT NORMAL FPR] (lower is better)")
            print(f"  PCA-only       : fpr={fpr_pca:.6f}")
            print(f"  GMM-only       : fpr={fpr_gmm:.6f}")
            print(f"  Combined(AND)  : fpr={fpr_and:.6f}")
            print(f"  Combined(OR)   : fpr={fpr_or:.6f}")

            # ============================================================
            # 7.2) FIXED UNSUPERVISED SELECTION (deterministic)
            # ============================================================
            # deterministic slice for Pandas collection (avoid huge memory)
            SLICE_MOD = 10  # keep ~10% deterministically; set 1 for full

            norm_scores_pca = (
                train_pca_normal_fit.filter((col("hid") % lit(SLICE_MOD)) == lit(0)).orderBy("hid").select(
                    "anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(float).values)

            unlab_scores_pca = (unlab_gmm.filter((col("hid") % lit(SLICE_MOD)) == lit(0)).orderBy("hid").select(
                "anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(float).values)

            norm_scores_gmm = (train_gmm_fit.filter((col("hid") % lit(SLICE_MOD)) == lit(0)).orderBy("hid").select(
                "anomaly_score_gmm").toPandas()["anomaly_score_gmm"].astype(float).values)

            unlab_scores_gmm = (unlab_gmm.filter((col("hid") % lit(SLICE_MOD)) == lit(0)).orderBy("hid").select(
                "anomaly_score_gmm").toPandas()["anomaly_score_gmm"].astype(float).values)

            norm_scores_comb = (train_gmm_fit.filter((col("hid") % lit(SLICE_MOD)) == lit(0)).orderBy("hid").select(
                (col("anomaly_score_pca") + col("anomaly_score_gmm")).alias("comb_score")).toPandas()[
                                    "comb_score"].astype(float).values)

            unlab_scores_comb = (unlab_gmm.filter((col("hid") % lit(SLICE_MOD)) == lit(0)).orderBy("hid").select(
                (col("anomaly_score_pca") + col("anomaly_score_gmm")).alias("comb_score")).toPandas()[
                                     "comb_score"].astype(float).values)

            sep_pca = separation_z(norm_scores_pca, unlab_scores_pca)
            sep_gmm = separation_z(norm_scores_gmm, unlab_scores_gmm)
            sep_comb = separation_z(norm_scores_comb, unlab_scores_comb)

            # deterministic stability splits (no random sample)
            u1 = unlab_gmm.filter((col("hid") % lit(100)) < lit(80))
            u2 = unlab_gmm.filter((col("hid") % lit(100)) >= lit(20))

            stab_pca = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_pca")
            stab_gmm = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_gmm")
            stab_and = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_and")
            stab_or = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_or")

            score_pca = (sep_pca * stab_pca) / (fpr_pca + eps)
            score_gmm = (sep_gmm * stab_gmm) / (fpr_gmm + eps)
            score_and = (sep_comb * stab_and) / (fpr_and + eps)
            score_or = (sep_comb * stab_or) / (fpr_or + eps)

            print("\n[UNSUPERVISED SCORE (separation * stability / FPR)]")
            print(f"  PCA          : sep={sep_pca:.4f},  stab={stab_pca:.4f}, fpr={fpr_pca:.6f}, score={score_pca:.6f}")
            print(f"  GMM          : sep={sep_gmm:.4f},  stab={stab_gmm:.4f}, fpr={fpr_gmm:.6f}, score={score_gmm:.6f}")
            print(f"  Combined(AND): sep={sep_comb:.4f}, stab={stab_and:.4f}, fpr={fpr_and:.6f}, score={score_and:.6f}")
            print(f"  Combined(OR) : sep={sep_comb:.4f}, stab={stab_or:.4f},  fpr={fpr_or:.6f},  score={score_or:.6f}")

            best_method = \
            max([("pca", score_pca), ("gmm", score_gmm), ("and", score_and), ("or", score_or)], key=lambda x: x[1])[0]

            print(f"\n[SELECTED] Best method (fixed unsupervised): {best_method}")

            if method in ["pca", "gmm", "and", "or"]:
                best_method = method  # user override

            if best_method == "pca":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_pca"))
            elif best_method == "gmm":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_gmm"))
            elif best_method == "and":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_and"))
            else:
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_or"))

            # ============================================================
            # 8) DEBUG ONLY: evaluation if you have true labels for unlabeled
            # ============================================================
            try:
                train_unlabeled_eval_df = unlab_gmm.join(
                    sequences_df.select(col(id_col), col("Label").alias("true_label")), on=id_col, how="inner")

                pdf_unlabeled = train_unlabeled_eval_df.select("true_label", "pseudo_label_pca", "pseudo_label_gmm",
                    "pseudo_label_and", "pseudo_label_or", "Final_Label").toPandas()

                print("\n=== Classification_report on unlabeled (PCA-only) ===")
                print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_pca"], digits=3))
                print("\n=== Classification_report on unlabeled (GMM-only) ===")
                print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_gmm"], digits=3))
                print("\n=== Classification_report on unlabeled (Combined-AND) ===")
                print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_and"], digits=3))
                print("\n=== Classification_report on unlabeled (Combined-OR) ===")
                print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_or"], digits=3))
                print("\n=== Classification_report on unlabeled (SELECTED Final_Label) ===")
                print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["Final_Label"], digits=3))
            except Exception as e:
                print(f"[DEBUG] Skipping debug evaluation (Label column missing or error): {e}")


            # ============================================================
            # 9) Build final training set using SELECTED pseudo labels
            # ============================================================

            # Columns you want for classifier
            keep_cols = [id_col, "pca_features", "Final_Label"]

            # ---- Normal training part (true normal = 0) ----
            # IMPORTANT: use the SAME PCA transform that produced unlabeled pca_features
            # In the updated pipeline this is train_pca_normal_fit (PCA trained on norm_fit_df)
            train_df_normal = (
                train_pca_normal_fit.withColumn("Final_Label", lit(0).cast("int")).select(*keep_cols))

            # ---- Unlabeled training part (pseudo labels from SELECTED method) ----
            train_df_unlabeled = (
                unlab_gmm.withColumn("Final_Label", col("Final_Label").cast("int")).select(*keep_cols))

            # ---- UNION ----
            df_final_train_cls = train_df_normal.unionByName(train_df_unlabeled, allowMissingColumns=False)

            # ============================================================
            # Prepare test/val
            # ============================================================
            # Ensure test_pca / val_pca come from SAME PCA model and have pca_features
            # If Label exists, keep it; otherwise set Final_Label to null.

            test_has_label = "Label" in df_test.columns
            val_has_label = "Label" in df_val.columns

            df_test_cls = (test_pca.withColumn("Final_Label",
                (col("Label").cast("int") if test_has_label else lit(None).cast("int"))).select(id_col,
                                                                                                "pca_features",
                                                                                                "Final_Label"))

            df_val_cls = (val_pca.withColumn("Final_Label",
                (col("Label").cast("int") if val_has_label else lit(None).cast("int"))).select(id_col,
                                                                                               "pca_features",
                                                                                               "Final_Label"))

            # ============================================================
            # FINAL TRAINING QUALITY REPORT (true Label vs Final_Label)
            # ============================================================

            # 1) Join true labels onto the final training set (only where Label exists)
            if "Label" not in sequences_df.columns:
                print(
                    "[WARN] sequences_df has no 'Label' column -> cannot compute final training classification report.")
            else:
                df_train_quality = (
                    df_final_train_cls.join(sequences_df.select(col(id_col), col("Label").alias("true_label")),
                        on=id_col, how="inner").select("true_label", "Final_Label").dropna())

                # 2) Ensure there is data to evaluate
                n_quality = df_train_quality.count()
                if n_quality == 0:
                    print("[WARN] No rows with both true_label and Final_Label -> report skipped.")
                else:
                    # 3) Convert to pandas for sklearn report
                    pdf_train_quality = df_train_quality.toPandas()
                    pdf_train_quality["true_label"] = pdf_train_quality["true_label"].astype(int)
                    pdf_train_quality["Final_Label"] = pdf_train_quality["Final_Label"].astype(int)

                    print(f"\n=== Classification_report on Full FINAL TRAIN SET (n={n_quality}) ===")
                    print(classification_report(pdf_train_quality["true_label"], pdf_train_quality["Final_Label"],
                        digits=3))


            # return for next stage
            return df_final_train_cls, df_test_cls, df_val_cls



            # start  good ----------------idea 2--------case 4---------------------------------------------------------
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
            # Settings
            # -----------------------------
            STABILITY_SEED1 = 21
            STABILITY_SEED2 = 99

            # -----------------------------
            # Helpers
            # -----------------------------
            def knee_threshold_from_scores(scores_np, min_quantile_floor=0.95, knee_margin=0.02,
                                           default_if_no_knee=0.99):
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

            def jaccard_anomaly_sets(df1, df2, id_col, label_col):
                a1 = df1.filter(col(label_col) == 1).select(id_col).distinct()
                a2 = df2.filter(col(label_col) == 1).select(id_col).distinct()
                inter = a1.join(a2, on=id_col, how="inner").count()
                union = a1.union(a2).distinct().count()
                return inter / union if union > 0 else 0.0

            eps = 1e-9

            def fbeta(p, r, beta=1.0):
                b2 = beta * beta
                return (1 + b2) * p * r / (b2 * p + r + eps)

            print("\n🧠 Using PCA + GMM for novelty detection ...")

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
            # 2) Fit PCA on NORMAL + transform NORMAL/UNLABELED/TEST/VAL
            # ============================================================
            pca = SparkPCA(k=best_k, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(train_normal_df)

            train_pca_normal = pca_model.transform(train_normal_df)
            train_unlabeled_pca = pca_model.transform(train_unlabeled_df)
            test_pca = pca_model.transform(df_test)
            val_pca = pca_model.transform(df_val)

            # ============================================================
            # 3) PCA reconstruction error -> anomaly_score_pca
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
                                                                 reconstruction_error(col(feature_col),
                                                                                      col("pca_features")))

            # ============================================================
            # 4) PCA threshold (knee on NORMAL)
            # ============================================================
            scores_pca_np = train_pca_normal.select("anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(
                float).values
            thr_pca, p_knee_pca, p_use_pca = knee_threshold_from_scores(scores_pca_np)
            print(f"[INFO] PCA p_knee≈{p_knee_pca:.4f}, using p={p_use_pca:.4f}, threshold={thr_pca:.6f}")

            train_unlabeled_pca = train_unlabeled_pca.withColumn("pseudo_label_pca",
                                                                 when(col("anomaly_score_pca") > lit(thr_pca),
                                                                      1).otherwise(0))

            # ============================================================
            # 5) Fit GMM on NORMAL in PCA space + score UNLABELED
            # ============================================================
            gmm_k = 10
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
            thr_gmm, p_knee_gmm, p_use_gmm = knee_threshold_from_scores(scores_gmm_np)
            print(f"[INFO] GMM(k={gmm_k}) p_knee≈{p_knee_gmm:.4f}, using p={p_use_gmm:.4f}, threshold={thr_gmm:.6f}")

            unlab_gmm = unlab_gmm.withColumn("pseudo_label_gmm",
                                             when(col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0))

            # ============================================================
            # 7) Create BOTH combined rules (AND + OR)
            # ============================================================
            unlab_gmm = unlab_gmm.withColumn("pseudo_label_and",
                                             when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1),
                                                  1).otherwise(0))
            unlab_gmm = unlab_gmm.withColumn("pseudo_label_or",
                                             when((col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1),
                                                  1).otherwise(0))

            # ============================================================
            # 7.1) Holdout-normal FPR computation (precision proxy)
            # ============================================================
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

            scores_pca_fit_np = norm_fit_pca.select("anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(
                float).values
            thr_pca_fit, _, _ = knee_threshold_from_scores(scores_pca_fit_np)
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
            thr_gmm_fit, _, _ = knee_threshold_from_scores(scores_gmm_fit_np)
            fpr_gmm = fpr_on_holdout(norm_hold_gmm, "anomaly_score_gmm", thr_gmm_fit)

            # combined FPRs on normals using flags
            norm_hold_join = (norm_hold_gmm.select(id_col, "anomaly_score_gmm").join(
                norm_hold_pca.select(id_col, "anomaly_score_pca"), on=id_col, how="inner").withColumn("pca_flag", when(
                col("anomaly_score_pca") > lit(thr_pca_fit), 1).otherwise(0)).withColumn("gmm_flag", when(
                col("anomaly_score_gmm") > lit(thr_gmm_fit), 1).otherwise(0)).withColumn("and_flag", when(
                (col("pca_flag") == 1) & (col("gmm_flag") == 1), 1).otherwise(0)).withColumn("or_flag", when(
                (col("pca_flag") == 1) | (col("gmm_flag") == 1), 1).otherwise(0)))

            fpr_and = anomaly_rate(norm_hold_join, "and_flag")
            fpr_or = anomaly_rate(norm_hold_join, "or_flag")

            print("\n[HOLDOUT NORMAL FPR] (lower is better)")
            print(f"  PCA-only       : fpr={fpr_pca:.6f}")
            print(f"  GMM-only       : fpr={fpr_gmm:.6f}")
            print(f"  Combined(AND)  : fpr={fpr_and:.6f}")
            print(f"  Combined(OR)   : fpr={fpr_or:.6f}")

            # ============================================================
            # 7.2) FIXED UNSUPERVISED SELECTION (no "anomaly-rate" recall proxy)
            #      Uses:
            #        - separation between normal/unlabeled score distributions
            #        - stability (Jaccard)
            #        - FPR on normal holdout
            # ============================================================

            eps = 1e-9

            def separation_z(norm_scores, unlab_scores):
                mu_n = float(np.mean(norm_scores))
                sd_n = float(np.std(norm_scores)) + eps
                mu_u = float(np.mean(unlab_scores))
                return (mu_u - mu_n) / sd_n

            # --- Collect scores (avoid huge collect if too big; sample if needed) ---
            SAMPLE_FRAC = 0.2  # if dataset huge, keep small sample; set 1.0 if manageable

            norm_scores_pca = (train_pca_normal.sample(False, SAMPLE_FRAC, 1).select("anomaly_score_pca").toPandas()[
                                   "anomaly_score_pca"].astype(float).values)

            unlab_scores_pca = (
                train_unlabeled_pca.sample(False, SAMPLE_FRAC, 2).select("anomaly_score_pca").toPandas()[
                    "anomaly_score_pca"].astype(float).values)

            norm_scores_gmm = (train_gmm.sample(False, SAMPLE_FRAC, 3).select("anomaly_score_gmm").toPandas()[
                                   "anomaly_score_gmm"].astype(float).values)

            unlab_scores_gmm = (unlab_gmm.sample(False, SAMPLE_FRAC, 4).select("anomaly_score_gmm").toPandas()[
                                    "anomaly_score_gmm"].astype(float).values)

            # --- Combined score for AND/OR:
            # Use a continuous score that doesn't explode with OR:
            # max(score_pca_normed, score_gmm_normed) is OK, but we’ll keep it simple:
            # score_comb = anomaly_score_pca + anomaly_score_gmm (more stable than max)
            unlab_scores_comb = (unlab_gmm.sample(False, SAMPLE_FRAC, 5).select(
                (col("anomaly_score_pca") + col("anomaly_score_gmm")).alias("comb_score")).toPandas()[
                                     "comb_score"].astype(float).values)

            norm_scores_comb = (train_gmm.sample(False, SAMPLE_FRAC, 6).select(
                (col("anomaly_score_pca") + col("anomaly_score_gmm")).alias("comb_score")).toPandas()[
                                    "comb_score"].astype(float).values)

            # --- separations ---
            sep_pca = separation_z(norm_scores_pca, unlab_scores_pca)
            sep_gmm = separation_z(norm_scores_gmm, unlab_scores_gmm)
            sep_comb = separation_z(norm_scores_comb, unlab_scores_comb)

            # --- stability (labels) ---
            u1 = unlab_gmm.sample(False, 0.8, STABILITY_SEED1)
            u2 = unlab_gmm.sample(False, 0.8, STABILITY_SEED2)

            stab_pca = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_pca")
            stab_gmm = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_gmm")
            stab_and = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_and")
            stab_or = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_or")

            # --- score: separation * stability / FPR ---
            score_pca = (sep_pca * stab_pca) / (fpr_pca + eps)
            score_gmm = (sep_gmm * stab_gmm) / (fpr_gmm + eps)

            # AND/OR share same separation (continuous), but differ in stability and FPR
            score_and = (sep_comb * stab_and) / (fpr_and + eps)
            score_or = (sep_comb * stab_or) / (fpr_or + eps)

            print("\n[UNSUPERVISED SCORE (separation * stability / FPR)]")
            print(f"  PCA          : sep={sep_pca:.4f},  stab={stab_pca:.4f}, fpr={fpr_pca:.6f}, score={score_pca:.6f}")
            print(f"  GMM          : sep={sep_gmm:.4f},  stab={stab_gmm:.4f}, fpr={fpr_gmm:.6f}, score={score_gmm:.6f}")
            print(f"  Combined(AND): sep={sep_comb:.4f}, stab={stab_and:.4f}, fpr={fpr_and:.6f}, score={score_and:.6f}")
            print(f"  Combined(OR) : sep={sep_comb:.4f}, stab={stab_or:.4f},  fpr={fpr_or:.6f},  score={score_or:.6f}")

            best_method = \
                max([("pca", score_pca), ("gmm", score_gmm), ("and", score_and), ("or", score_or)], key=lambda x: x[1])[
                    0]

            print(f"\n[SELECTED] Best method (fixed unsupervised): {best_method}")

            if best_method == "pca":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_pca"))
            elif best_method == "gmm":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_gmm"))
            elif best_method == "and":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_and"))
            else:
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_or"))

            # ============================================================
            # 8) DEBUG ONLY: evaluation if you have true labels for unlabeled
            # ============================================================
            train_unlabeled_eval_df = unlab_gmm.join(sequences_df.select(col(id_col), col("Label").alias("true_label")),
                                                     on=id_col, how="inner")

            pdf_unlabeled = train_unlabeled_eval_df.select("true_label", "pseudo_label_pca", "pseudo_label_gmm",
                                                           "pseudo_label_and", "pseudo_label_or",
                                                           "Final_Label").toPandas()

            print("\n=== Classification_report on unlabeled (PCA-only) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_pca"], digits=3))
            print("\n=== Classification_report on unlabeled (GMM-only) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_gmm"], digits=3))
            print("\n=== Classification_report on unlabeled (Combined-AND) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_and"], digits=3))
            print("\n=== Classification_report on unlabeled (Combined-OR) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["pseudo_label_or"], digits=3))
            print("\n=== Classification_report on unlabeled (SELECTED Final_Label) ===")
            print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["Final_Label"], digits=3))

            #exit()

            # ============================================================
            # 9) Build final training set using SELECTED pseudo labels
            # ============================================================
            # Columns you want for classifier
            keep_cols = [id_col, "pca_features", "Final_Label"]

            # ---- Normal training part (true normal = 0) ----
            train_df_normal = (train_pca_normal.withColumn("Final_Label", lit(0).cast("int")).select(*keep_cols))

            # ---- Unlabeled training part (pseudo labels from SELECTED method) ----
            train_df_unlabeled = (
                unlab_gmm.withColumn("Final_Label", col("Final_Label").cast("int")).select(*keep_cols))

            # ---- UNION (this is correct) ----
            df_final_train_cls = train_df_normal.unionByName(train_df_unlabeled, allowMissingColumns=False)

            # ---- Prepare test/val (use true Label if exists; else just keep pca_features + id) ----
            df_test_cls = (test_pca.withColumn("Final_Label", col("Label").cast("int"))  # if test has Label
                           .select(id_col, "pca_features", "Final_Label"))

            df_val_cls = (val_pca.withColumn("Final_Label", col("Label").cast("int"))  # if val has Label
                          .select(id_col, "pca_features", "Final_Label"))

            # Join back the true label from sequences_df (or from the original train source)
            df_train_quality = (
                df_final_train_cls.join(sequences_df.select(col(id_col), col("Label").alias("true_label")), on=id_col,
                                        how="inner").select("true_label", "Final_Label").dropna())

            # Convert to pandas for sklearn report
            pdf_train_quality = df_train_quality.toPandas()
            pdf_train_quality["true_label"] = pdf_train_quality["true_label"].astype(int)
            pdf_train_quality["Final_Label"] = pdf_train_quality["Final_Label"].astype(int)

            print("\n=== Classification_report on FINAL TRAIN SET (true_label vs Final_Label) ===")
            print(classification_report(pdf_train_quality["true_label"], pdf_train_quality["Final_Label"], digits=3))
            print(f"\n[SELECTED] Best method (fixed unsupervised): {best_method}")


            # return for next stage
            return df_final_train_cls, df_test_cls, df_val_cls

            '''
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


            return df_final_train_cls, df_test_cls ,df_val_cls
            '''
