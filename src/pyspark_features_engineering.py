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

        # --- split (use your Temp_label) ---
        train_df = summ_train_test_val_combine.filter(F.col("Temp_label").isin([0, 999]))
        test_df = summ_train_test_val_combine.filter(F.col("Temp_label") == 888)
        val_df = summ_train_test_val_combine.filter(F.col("Temp_label") == 777)

        # --- fit ONLY on train ---
        scaler = StandardScaler(inputCol="features_vec", outputCol="features_vec_final", withMean=True, withStd=True)
        scaler_model = scaler.fit(train_df)

        # --- transform all using the same scaler_model ---
        train_scaled = scaler_model.transform(train_df)
        test_scaled = scaler_model.transform(test_df)
        val_scaled = scaler_model.transform(val_df)

        print("✅ StandardScaler fit on TRAIN only, applied to train/test/val (no leakage)")
        print('scaling done finally -------')

        # If you still want a single combined DF (same as before):
        summ_train_test_val_combine_scaled = train_scaled.unionByName(test_scaled).unionByName(val_scaled)

        # Keep your existing outputs:
        X_sequences_df = summ_train_test_val_combine_scaled.select("features_vec_final")
        y_sequences_df = summ_train_test_val_combine_scaled.select("Label")

        return summ_train_test_val_combine_scaled, X_sequences_df, y_sequences_df

        '''
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
        '''


    @staticmethod
    def novelty_detection_label_establishmentxxxxx(sequences_df, spark, method="auto", feature_col="features_vec_final",
            id_col="Node_block_id", TARGET_FPR=0.01, candidate_ks=(10, 20, 40, 50, 60, 70), target_variance=0.999,
            gmm_ks=(2, 4, 6, 8, 10), SEED=123, SLICE_MOD=10, print_reports=True):
        # ============================================================
        # 0) Inspect + Robust label normalization
        # ============================================================
        sequences_df.printSchema()
        sequences_df.groupBy("Label").count().show()
        sequences_df.groupBy("Temp_label").count().show()

        # Robust label normalization -> numeric 0/1
        sequences_df = sequences_df.withColumn("Label_str", F.lower(F.trim(F.col("Label").cast("string"))))
        sequences_df = sequences_df.withColumn("Label",
            F.when(F.col("Label_str").isin("normal", "0"), F.lit(0)).when(F.col("Label_str").isin("anomaly", "1"),
                                                                          F.lit(1)).otherwise(
                F.lit(None).cast("int"))).drop("Label_str")

        bad_cnt = sequences_df.filter(F.col("Label").isNull()).count()
        if bad_cnt > 0:
            print("[ERROR] Unknown label values found after normalization. Sample:")
            sequences_df.filter(F.col("Label").isNull()).select(id_col, "Label").show(50, False)
            raise ValueError("Unknown label values after normalization. Fix label mapping upstream.")

        print(f"\n🚀 Starting novelty detection using method = {str(method).upper()}")

        # ============================================================
        # 1) Split by Temp_label
        # ============================================================
        train_normal_df = sequences_df.filter(col("Temp_label") == 0)
        train_unlabeled_df = sequences_df.filter(col("Temp_label") == 999)
        df_test = sequences_df.filter(col("Temp_label") == 888)
        df_val = sequences_df.filter(col("Temp_label") == 777)

        print('info label  ..........................................')
        train_normal_df.groupBy("Label").count().show()
        train_unlabeled_df.groupBy("Label").count().show()
        df_test.groupBy("Label").count().show()
        df_val.groupBy("Label").count().show()

        if train_normal_df.count() == 0:
            raise ValueError("❌ No normal logs (Temp_label=0) found for training.")
        if train_unlabeled_df.count() == 0:
            raise ValueError("❌ No unlabeled logs (Temp_label=999) found for novelty detection.")

        eps = 1e-9

        # ============================================================
        # 2) Deterministic key + deterministic normal split (80/20)
        # ============================================================
        train_normal_df = train_normal_df.withColumn("hid", F.xxhash64(col(id_col)))
        train_unlabeled_df = train_unlabeled_df.withColumn("hid", F.xxhash64(col(id_col)))
        df_test = df_test.withColumn("hid", F.xxhash64(col(id_col)))
        df_val = df_val.withColumn("hid", F.xxhash64(col(id_col)))

        norm_fit_df = train_normal_df.filter((col("hid") % lit(100)) < lit(80))
        norm_holdout_df = train_normal_df.filter((col("hid") % lit(100)) >= lit(80))

        # ============================================================
        # Helpers
        # ============================================================
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

        def jaccard_anomaly_sets(df1, df2, id_col_local, label_col_local):
            a1 = df1.filter(col(label_col_local) == 1).select(id_col_local).distinct()
            a2 = df2.filter(col(label_col_local) == 1).select(id_col_local).distinct()
            inter = a1.join(a2, on=id_col_local, how="inner").count()
            union = a1.union(a2).distinct().count()
            return inter / union if union > 0 else 0.0

        def separation_z(norm_scores, unlab_scores):
            mu_n = float(np.mean(norm_scores))
            sd_n = float(np.std(norm_scores)) + eps
            mu_u = float(np.mean(unlab_scores))
            return max(0.0, (mu_u - mu_n) / sd_n)

        print("\n🧠 Using PCA + GMM for novelty detection (updated/stable) ...")

        # ============================================================
        # 3) Choose PCA k on NORMAL-fit only
        # ============================================================
        best_k = None
        for k in candidate_ks:
            print(f"[INFO] Testing PCA with k={k}")
            pca_tmp = SparkPCA(k=int(k), inputCol=feature_col, outputCol=f"pca_features_k{k}")
            pca_tmp_model = pca_tmp.fit(norm_fit_df)
            explained_variance = float(sum(pca_tmp_model.explainedVariance))
            print(f"[INFO] PCA k={k}, cumulative explained variance = {explained_variance:.6f}")
            if explained_variance >= float(target_variance):
                best_k = int(k)
                print(f"[SELECTED] First k reaching target variance: {best_k}")
                break

        if best_k is None:
            best_k = int(candidate_ks[-1])
            print(f"[WARNING] Target variance not reached. Using max k = {best_k}")

        print(f"[RESULT] Selected PCA components (best_k): {best_k}")

        # ============================================================
        # 4) Fit PCA on NORMAL-fit + transform all
        # ============================================================
        pca = SparkPCA(k=int(best_k), inputCol=feature_col, outputCol="pca_features")
        pca_model = pca.fit(norm_fit_df)

        train_pca_normal_fit = pca_model.transform(norm_fit_df)
        train_pca_normal_hold = pca_model.transform(norm_holdout_df)
        train_unlabeled_pca = pca_model.transform(train_unlabeled_df)
        test_pca = pca_model.transform(df_test)
        val_pca = pca_model.transform(df_val)

        # ============================================================
        # 5) PCA reconstruction error
        # ============================================================
        pc = pca_model.pc.toArray()
        d = len(train_pca_normal_fit.select(feature_col).head()[0])
        use_pc_dk = (pc.shape[0] == d)

        pc_b = spark.sparkContext.broadcast(pc)
        use_pc_dk_b = spark.sparkContext.broadcast(use_pc_dk)

        @F.udf(DoubleType())
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
                                                               reconstruction_error(col(feature_col),
                                                                                    col("pca_features")))
        train_pca_normal_hold = train_pca_normal_hold.withColumn("anomaly_score_pca",
                                                                 reconstruction_error(col(feature_col),
                                                                                      col("pca_features")))
        train_unlabeled_pca = train_unlabeled_pca.withColumn("anomaly_score_pca", reconstruction_error(col(feature_col),
                                                                                                       col("pca_features")))
        test_pca = test_pca.withColumn("anomaly_score_pca", reconstruction_error(col(feature_col), col("pca_features")))
        val_pca = val_pca.withColumn("anomaly_score_pca", reconstruction_error(col(feature_col), col("pca_features")))

        # ============================================================
        # 6) PCA threshold + pseudo labels
        # ============================================================
        thr_pca = threshold_from_norm_fit(train_pca_normal_fit, "anomaly_score_pca", target_fpr=TARGET_FPR)
        fpr_pca = fpr_on_holdout(train_pca_normal_hold, "anomaly_score_pca", thr_pca)
        print(f"[PCA] thr={thr_pca:.6f}, holdout FPR={fpr_pca:.6f}")

        train_unlabeled_pca = train_unlabeled_pca.withColumn("pseudo_label_pca",
                                                             when(col("anomaly_score_pca") > lit(thr_pca), 1).otherwise(
                                                                 0))

        # ============================================================
        # 7) Fit GMM on NORMAL-fit PCA space (choose k by BIC), score by NLL
        # ============================================================
        d_pca = int(best_k)

        def bic_from_model(model, df_features, k, d_dim):
            try:
                ll = float(model.summary.logLikelihood)
            except Exception:
                return float("inf")
            n = df_features.count()
            p = (k - 1) + (k * d_dim) + (k * (d_dim * (d_dim + 1) // 2))
            return -2.0 * ll + p * np.log(max(n, 1))

        feat_df = train_pca_normal_fit.select("pca_features")

        best_gmm_model = None
        best_bic = float("inf")
        best_gmm_k = None

        for k in gmm_ks:
            k = int(k)
            gmm = GaussianMixture(k=k, seed=int(SEED), featuresCol="pca_features", predictionCol="gmm_cluster",
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
        means = np.stack([vec_to_np(g.mean) for g in gaussians], axis=0)
        covs = np.stack([cov_to_2d(g.cov, d_pca) for g in gaussians], axis=0)

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

        @F.udf(DoubleType())
        def gmm_nll(pca_vec):
            z = np.array(pca_vec.toArray(), dtype=float)
            P = bc_params.value
            w = P["weights"]
            m = P["means"]
            ic = P["inv_covs"]
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
            return float(-logp)

        train_gmm_fit = train_pca_normal_fit.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
        train_gmm_hold = train_pca_normal_hold.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
        unlab_gmm = train_unlabeled_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))

        test_pca = test_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
        val_pca = val_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))

        # ============================================================
        # 8) GMM threshold + pseudo labels
        # ============================================================
        thr_gmm = threshold_from_norm_fit(train_gmm_fit, "anomaly_score_gmm", target_fpr=TARGET_FPR)
        fpr_gmm = fpr_on_holdout(train_gmm_hold, "anomaly_score_gmm", thr_gmm)
        print(f"[GMM-NLL] thr={thr_gmm:.6f}, holdout FPR={fpr_gmm:.6f}")

        unlab_gmm = unlab_gmm.withColumn("pseudo_label_gmm",
                                         when(col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0))

        # ============================================================
        # 9) Combine rules
        # ============================================================
        unlab_gmm = (unlab_gmm.withColumn("pseudo_label_and",
                                          when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1),
                                               1).otherwise(0)).withColumn("pseudo_label_or", when(
            (col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1), 1).otherwise(0)))

        # ============================================================
        # 9.1) Holdout normal FPR for combined rules
        # ============================================================
        norm_hold_join = (train_gmm_hold.select(id_col, "anomaly_score_gmm").join(
            train_pca_normal_hold.select(id_col, "anomaly_score_pca"), on=id_col, how="inner").withColumn("pca_flag",
                                                                                                          when(
                                                                                                              col("anomaly_score_pca") > lit(
                                                                                                                  thr_pca),
                                                                                                              1).otherwise(
                                                                                                              0)).withColumn(
            "gmm_flag", when(col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0)).withColumn("and_flag", when(
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
        # 9.2) Unsupervised selection score
        # ============================================================
        norm_scores_pca = (
            train_pca_normal_fit.filter((col("hid") % lit(int(SLICE_MOD))) == lit(0)).orderBy("hid").select(
                "anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(float).values)
        unlab_scores_pca = (unlab_gmm.filter((col("hid") % lit(int(SLICE_MOD))) == lit(0)).orderBy("hid").select(
            "anomaly_score_pca").toPandas()["anomaly_score_pca"].astype(float).values)

        norm_scores_gmm = (train_gmm_fit.filter((col("hid") % lit(int(SLICE_MOD))) == lit(0)).orderBy("hid").select(
            "anomaly_score_gmm").toPandas()["anomaly_score_gmm"].astype(float).values)
        unlab_scores_gmm = (unlab_gmm.filter((col("hid") % lit(int(SLICE_MOD))) == lit(0)).orderBy("hid").select(
            "anomaly_score_gmm").toPandas()["anomaly_score_gmm"].astype(float).values)

        norm_scores_comb = (train_gmm_fit.filter((col("hid") % lit(int(SLICE_MOD))) == lit(0)).orderBy("hid").select(
            (col("anomaly_score_pca") + col("anomaly_score_gmm")).alias("comb_score")).toPandas()["comb_score"].astype(
            float).values)
        unlab_scores_comb = (unlab_gmm.filter((col("hid") % lit(int(SLICE_MOD))) == lit(0)).orderBy("hid").select(
            (col("anomaly_score_pca") + col("anomaly_score_gmm")).alias("comb_score")).toPandas()["comb_score"].astype(
            float).values)

        sep_pca = separation_z(norm_scores_pca, unlab_scores_pca)
        sep_gmm = separation_z(norm_scores_gmm, unlab_scores_gmm)
        sep_comb = separation_z(norm_scores_comb, unlab_scores_comb)

        u1 = unlab_gmm.filter((col("hid") % lit(100)) < lit(80))
        u2 = unlab_gmm.filter((col("hid") % lit(100)) >= lit(80))

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
        if isinstance(method, str) and method.lower() in ["pca", "gmm", "and", "or"]:
            best_method = method.lower()

        print(f"\n[SELECTED] Best method: {best_method}")

        if best_method == "pca":
            unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_pca"))
        elif best_method == "gmm":
            unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_gmm"))
        elif best_method == "and":
            unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_and"))
        else:
            unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_or"))

        # ============================================================
        # 10) Build final training set WITHOUT confidence filtering (Option A)
        #     ✅ keep original vector until the end
        # ============================================================
        keep_cols_train = [id_col, feature_col,  # ✅ original vector kept
            "pca_features", "anomaly_score_pca", "anomaly_score_gmm", "Final_Label", "hid"]

        train_df_normal_cls = (
            train_pca_normal_fit.withColumn("Final_Label", lit(0).cast("int")).select(*keep_cols_train))

        train_df_unlabeled_cls = (
            unlab_gmm.withColumn("Final_Label", col("Final_Label").cast("int")).select(*keep_cols_train))

        df_final_train_cls = train_df_normal_cls.unionByName(train_df_unlabeled_cls, allowMissingColumns=False)

        print("\n[CHECK] Final train class balance (NO confidence filtering):")
        df_final_train_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()

        # ============================================================
        # 11) Prepare test/val (true labels)
        # ============================================================
        keep_cols_eval = [id_col, feature_col, "pca_features", "anomaly_score_pca", "anomaly_score_gmm", "Final_Label"]

        df_test_cls = (test_pca.withColumn("Final_Label", col("Label").cast("int")).select(*keep_cols_eval))

        df_val_cls = (val_pca.withColumn("Final_Label", col("Label").cast("int")).select(*keep_cols_eval))

        n_null_test = df_test_cls.filter(col("Final_Label").isNull()).count()
        n_null_val = df_val_cls.filter(col("Final_Label").isNull()).count()
        if n_null_test > 0:
            raise ValueError(f"[ERROR] test Final_Label has {n_null_test} NULLs.")
        if n_null_val > 0:
            raise ValueError(f"[ERROR] val Final_Label has {n_null_val} NULLs.")

        print("\n[CHECK] Val/Test class balance:")
        df_val_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()
        df_test_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()

        # ============================================================
        # 12) REPORTS (NO report_max_rows) - requested by you
        # ============================================================
        if print_reports:
            print("\n==============================")
            print("  TRAIN QUALITY REPORTS")
            print("==============================")

            # (A) Unlabeled train only: true Label vs Final_Label
            try:
                df_unlabeled_quality = (
                    train_df_unlabeled_cls.join(sequences_df.select(col(id_col), col("Label").alias("true_label")),
                        on=id_col, how="inner").dropna())

                n_unlab = df_unlabeled_quality.count()
                if n_unlab > 0:
                    pdf_unlab = df_unlabeled_quality.select(col("true_label").cast("int").alias("y_true"),
                        col("Final_Label").cast("int").alias("y_pred")).toPandas()

                    print(f"\n=== Classification Report: UNLABELED TRAIN (Temp_label=999) (n={n_unlab}) ===")
                    print(classification_report(pdf_unlab["y_true"], pdf_unlab["y_pred"], digits=3))
                else:
                    print("\n[WARN] No rows available for unlabeled train report.")
            except Exception as e:
                print(f"[ERROR] Unlabeled train report failed: {e}")

            # (B) Combined train: true Label vs Final_Label
            try:
                df_full_quality = (
                    df_final_train_cls.join(sequences_df.select(col(id_col), col("Label").alias("true_label")),
                        on=id_col, how="inner").dropna())

                n_full = df_full_quality.count()
                if n_full > 0:
                    pdf_full = df_full_quality.select(col("true_label").cast("int").alias("y_true"),
                        col("Final_Label").cast("int").alias("y_pred")).toPandas()

                    print(f"\n=== Nayef - Classification Report: FULL TRAIN (NORMAL + UNLABELED) (n={n_full}) ===")
                    print(classification_report(pdf_full["y_true"], pdf_full["y_pred"], digits=3))
                else:
                    print("\n[WARN] No rows available for full train report.")
            except Exception as e:
                print(f"[ERROR] Full train report failed: {e}")

        # Drop hid from outputs (clean outputs for downstream)
        df_final_train_cls = df_final_train_cls.drop("hid")
        exit()


        return df_final_train_cls, df_test_cls, df_val_cls

    @staticmethod
    def novelty_detection_label_establishment(sequences_df: DataFrame, spark: SparkSession, method: str = "gmm"
                                              ):

        '''
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
        '''

        # --------------------------------------------------------------------------------
        # Ensure feature column is vector type
        if method.lower() == "gmm":

            from pyspark.sql.functions import col
            from pyspark.sql import functions as F
            from pyspark.sql.functions import col, lit, when
            from pyspark.ml.functions import vector_to_array
            from pyspark.ml.classification import LogisticRegression
            from pyspark.ml.evaluation import BinaryClassificationEvaluator
            from sklearn.metrics import precision_recall_curve
            from pyspark.sql.functions import when, lower, trim, col


            #------------(6)

            # ============================================
            # IsolationForest using SynapseML
            # ============================================

            from pyspark.sql import functions as F
            from pyspark.sql.functions import col, when
            from synapse.ml.isolationforest import IsolationForest
            from pyspark.sql.functions import col
            from pyspark.ml.functions import array_to_vector

            # -----------------------------
            # CONFIG
            # -----------------------------
            CONTAMINATION = 0.05
            SEED = 42

            # -----------------------------
            # SPLIT DATA
            # -----------------------------
            sequences_df = sequences_df.withColumn("features", array_to_vector(col("features")))

            train_normal_df = sequences_df.filter(col("Temp_label") == 0).cache()
            train_unlabeled_df = sequences_df.filter(col("Temp_label") == 999).cache()
            full_train_df = sequences_df.filter(col("Temp_label").isin([0, 999])).cache()

            if train_normal_df.count() == 0:
                raise ValueError("No normal data found.")

            # -----------------------------
            # TRAIN IF ON NORMAL ONLY
            # -----------------------------

            iso = IsolationForest(featuresCol="features", scoreCol="if_score", predictionCol="if_pred",
                contamination=float(CONTAMINATION), numEstimators=200, randomSeed=SEED)

            model = iso.fit(train_normal_df.select("features"))

            print("✅ SynapseML IsolationForest trained.")

            # -----------------------------
            # LABEL UNLABELED
            # -----------------------------
            unl_labeled = model.transform(train_unlabeled_df).withColumn("pseudo_label", col("if_pred").cast("int"))

            # -----------------------------
            # LABEL FULL TRAIN
            # -----------------------------
            full_labeled = model.transform(full_train_df) \
                .withColumn("pseudo_label", col("if_pred").cast("int"))

            # -----------------------------
            # DISTRIBUTION CHECK
            # -----------------------------
            print("\n--- UNLABELED pseudo_label distribution ---")
            unl_labeled.groupBy("pseudo_label").count().show()

            print("\n--- FULL TRAIN pseudo_label distribution ---")
            full_labeled.groupBy("pseudo_label").count().show()

            # -----------------------------
            # CLASSIFICATION REPORT
            # -----------------------------
            if "Label" in sequences_df.columns:

                def safe_div(a, b):
                    return float(a) / float(b) if b else 0.0

                def print_report(df, title):

                    agg \
                    = df.select(
                        col("Label").cast("int").alias("y"),
                        col("pseudo_label").cast("int").alias("p")
                    ).agg(
                        F.sum(((col("y") == 1) & (col("p") == 1)).cast("int")).alias("tp"),
                        F.sum(((col("y") == 0) & (col("p") == 1)).cast("int")).alias("fp"),
                        F.sum(((col("y") == 0) & (col("p") == 0)).cast("int")).alias("tn"),
                        F.sum(((col("y") == 1) & (col("p") == 0)).cast("int")).alias("fn"),
                        F.count(F.lit(1)).alias("n")
                    ).collect()[0]

                    tp, fp, tn, fn, n = [int(agg[k]) for k in ["tp","fp","tn","fn","n"]]

                    p1 = safe_div(tp, tp +fp)
                    r1 = safe_div(tp, tp +fn)
                    f1 = safe_div( 2 *p1 *r1, p1 +r1)

                    p0 = safe_div(tn, tn +fn)
                    r0 = safe_div(tn, tn +fp)
                    f0 = safe_div( 2 *p0 *r0, p0 +r0)

                    acc = safe_div(tp +tn, n)

                    print("\n" + "= " *80)
                    print(title)
                    print("= " *80)
                    print("class | precision | recall | f1-score")
                    print(f"0     | {p0:.4f} | {r0:.4f} | {f0:.4f}")
                    print(f"1     | {p1:.4f} | {r1:.4f} | {f1:.4f}")
                    print(f"accuracy: {acc:.4f}")
                    print(f"confusion matrix: tn={tn}, fp={fp}, fn={fn}, tp={tp}")

                print_report(unl_labeled, "📌 train_unlabeled_df Report")
                print_report(full_labeled, "📌 FULL TRAIN Report")

            else:
                print("⚠️ No ground-truth Label column found.")
            exit()
            # ============================================
            # OUTPUT DATAFRAMES:
            #   unl_labeled  -> unlabeled data with pseudo_label
            #   full_labeled -> full train with pseudo_label
            # ============================================

            #------------(5)
            from pyspark.sql.functions import col, when
            from pyspark.ml.iforest import IsolationForest

            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            train_unlabeled_df = sequences_df.filter(col("Temp_label") == 999)
            df_val = sequences_df.filter(col("Temp_label") == 777)
            df_test = sequences_df.filter(col("Temp_label") == 888)

            iso = IsolationForest(featuresCol="features", anomalyScoreCol="if_score", predictionCol="if_pred",
                contamination=0.05,  # tune later
                numTrees=200, maxDepth=10, seed=42)

            model = iso.fit(train_normal_df.select("features"))

            val_scored = model.transform(df_val)
            test_scored = model.transform(df_test)

            # If you want your own threshold instead of if_pred:
            thr = val_scored.approxQuantile("if_score", [0.95], 0.001)[0]  # top 5% as anomalies
            test_scored = test_scored.withColumn("novelty_pred", when(col("if_score") >= thr, 1).otherwise(0))


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
            # Map true labels to ints to match pseudo labels (0=normal, 1=anomaly)
            train_unlabeled_eval_df = (
                unlab_gmm.join(sequences_df.select(col(id_col), col("Label").alias("true_label_raw")), on=id_col,
                               how="inner").withColumn("true_label", when(
                    lower(trim(col("true_label_raw"))).isin("anomaly", "1", "true", "yes"), 1).when(
                    lower(trim(col("true_label_raw"))).isin("normal", "0", "false", "no"), 0).otherwise(None)).drop(
                    "true_label_raw"))

            pdf_unlabeled = (
                train_unlabeled_eval_df.select("true_label", "pseudo_label_pca", "pseudo_label_gmm", "pseudo_label_and",
                                               "pseudo_label_or", "Final_Label").dropna().toPandas())

            # Make sure everything is int
            for c in ["true_label", "pseudo_label_pca", "pseudo_label_gmm", "pseudo_label_and", "pseudo_label_or",
                      "Final_Label"]:
                pdf_unlabeled[c] = pdf_unlabeled[c].astype(int)

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

            # exit()

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
                df_final_train_cls.join(sequences_df.select(col(id_col), col("Label").alias("true_label_raw")),
                                        on=id_col, how="inner").withColumn("true_label",
                    when(lower(trim(col("true_label_raw"))).isin("anomaly", "1", "true", "yes"), 1).when(
                        lower(trim(col("true_label_raw"))).isin("normal", "0", "false", "no"), 0).otherwise(
                        None)).select("true_label", "Final_Label").dropna())

            pdf_train_quality = df_train_quality.toPandas()
            pdf_train_quality["true_label"] = pdf_train_quality["true_label"].astype(int)
            pdf_train_quality["Final_Label"] = pdf_train_quality["Final_Label"].astype(int)

            print("\n=== Classification_report on FINAL TRAIN SET (true_label vs Final_Label) ===")
            print(classification_report(pdf_train_quality["true_label"], pdf_train_quality["Final_Label"], digits=3))
            print(f"\n[SELECTED] Best method (fixed unsupervised): {best_method}")
            exit()

            #------------(4)

            # 0) Split data---- Good but not perfect in classification ---- (1)
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
            candidate_ks = [10, 20, 40, 50, 60, 70]
            # candidate_ks = [10, 20, 40, 50 ]
            # candidate_ks = [ 50, 60, 70, 80,90]
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
                                                                   reconstruction_error(col(feature_col),
                                                                                        col("pca_features")))
            train_pca_normal_hold = train_pca_normal_hold.withColumn("anomaly_score_pca",
                                                                     reconstruction_error(col(feature_col),
                                                                                          col("pca_features")))
            train_unlabeled_pca = train_unlabeled_pca.withColumn("anomaly_score_pca",
                                                                 reconstruction_error(col(feature_col),
                                                                                      col("pca_features")))

            # ============================================================
            # 4) PCA threshold (FPR-controlled)
            # ============================================================
            thr_pca = threshold_from_norm_fit(train_pca_normal_fit, "anomaly_score_pca", target_fpr=TARGET_FPR)
            fpr_pca = fpr_on_holdout(train_pca_normal_hold, "anomaly_score_pca", thr_pca)
            print(f"[PCA] thr={thr_pca:.6f}, holdout FPR={fpr_pca:.6f}")

            train_unlabeled_pca = train_unlabeled_pca.withColumn("pseudo_label_pca",
                                                                 when(col("anomaly_score_pca") > lit(thr_pca),
                                                                      1).otherwise(0))

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
                                             when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1),
                                                  1).otherwise(0))
            unlab_gmm = unlab_gmm.withColumn("pseudo_label_or",
                                             when((col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1),
                                                  1).otherwise(0))

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
                max([("pca", score_pca), ("gmm", score_gmm), ("and", score_and), ("or", score_or)], key=lambda x: x[1])[
                    0]

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
            train_df_normal = (train_pca_normal_fit.withColumn("Final_Label", lit(0).cast("int")).select(*keep_cols))

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

            df_test_cls = (test_pca.withColumn("Final_Label", (
                col("Label").cast("int") if test_has_label else lit(None).cast("int"))).select(id_col, "pca_features",
                                                                                               "Final_Label"))

            df_val_cls = (val_pca.withColumn("Final_Label", (
                col("Label").cast("int") if val_has_label else lit(None).cast("int"))).select(id_col, "pca_features",
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

                    print(f"\n=== Nayef : Classification_report on Full FINAL TRAIN SET (n={n_quality}) ===")
                    print(classification_report(pdf_train_quality["true_label"], pdf_train_quality["Final_Label"],
                                                digits=3))



            exit()


            #------------- (3)
            '''
            # -----------------------------
            # SETTINGS
            # -----------------------------
            METHOD = "auto"  # "auto" or force: "pca","gmm","and","or"
            TARGET_FPR = 0.01
            SEED = 123
            eps = 1e-9

            feature_col = "features_vec_final"
            id_col = "Node_block_id"
            temp_col = "Temp_label"

            # confidence filtering margins (increase to be stricter / less noisy)
            MARGIN_PCA = 0.25
            MARGIN_GMM = 0.25
            MARGIN_COMB = 0.20

            # cap pseudo anomalies to top X% most confident (reduces anomaly bias)
            CAP_PSEUDO_ANOM_TOP_PCT = 0.20  # keep only top 20% pseudo anomalies by comb_score; set None to disable

            # GBT params (good starting point)
            GBT_MAX_ITER = 120
            GBT_MAX_DEPTH = 5
            GBT_STEP_SIZE = 0.1

            # limit pandas pulls for reports (avoid OOM)
            PANDAS_CAP = 200000

            # -----------------------------
            # HELPERS
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

            def to_pandas_report(sdf, true_col, pred_col, cap=PANDAS_CAP):
                tmp = sdf.select(true_col, pred_col).dropna()
                n = tmp.count()
                if n == 0:
                    return None
                if n > cap:
                    # deterministic downsample using hash
                    tmp = tmp.withColumn("_h", F.xxhash64(
                        F.concat_ws("::", col(true_col).cast("string"), col(pred_col).cast("string"))))
                    k = int(np.ceil(n / cap))
                    tmp = tmp.filter((col("_h") % lit(k)) == lit(0)).drop("_h")
                pdf = tmp.toPandas()
                pdf[true_col] = pdf[true_col].astype(int)
                pdf[pred_col] = pdf[pred_col].astype(int)
                return pdf

            def tune_threshold_on_val(model, df_val_with_label, features_col=feature_col, label_col="label"):
                pred_val = model.transform(df_val_with_label).select(col(label_col).cast("int").alias("y"),
                    vector_to_array(col("probability")).getItem(1).alias("p1"))
                pdf_val = pred_val.toPandas()
                y = pdf_val["y"].astype(int).values
                p = pdf_val["p1"].astype(float).values

                prec, rec, thr = precision_recall_curve(y, p)
                f1 = (2 * prec * rec) / (prec + rec + 1e-12)

                # thr length = len(prec)-1; align safely
                best_i = int(np.nanargmax(f1))
                if best_i <= 0:
                    best_thr = 0.5
                else:
                    best_thr = float(thr[best_i - 1])

                return best_thr

            # ============================================================
            # 0) Robust label normalization
            # ============================================================
            sequences_df = sequences_df.withColumn("Label_str", F.lower(F.trim(F.col("Label").cast("string"))))
            sequences_df = sequences_df.withColumn("Label",
                F.when(F.col("Label_str").isin("normal", "0"), F.lit(0)).when(F.col("Label_str").isin("anomaly", "1"),
                                                                              F.lit(1)).otherwise(
                    F.lit(None).cast("int"))).drop("Label_str")

            bad = sequences_df.filter(F.col("Label").isNull()).count()
            if bad > 0:
                sequences_df.filter(F.col("Label").isNull()).select("Label").show(50, False)
                raise ValueError("Unknown label values after normalization.")

            print("\n[INFO] Label distribution:")
            sequences_df.groupBy("Label").count().show()
            print("[INFO] Temp_label distribution:")
            sequences_df.groupBy(temp_col).count().show()

            # ============================================================
            # 1) Split
            # ============================================================
            train_normal_df = sequences_df.filter(col(temp_col) == 0)
            train_unlabeled_df = sequences_df.filter(col(temp_col) == 999)
            df_test = sequences_df.filter(col(temp_col) == 888)
            df_val = sequences_df.filter(col(temp_col) == 777)

            print("\n[INFO] Split label breakdown:")
            train_normal_df.groupBy("Label").count().show()
            train_unlabeled_df.groupBy("Label").count().show()
            df_test.groupBy("Label").count().show()
            df_val.groupBy("Label").count().show()

            if train_normal_df.count() == 0:
                raise ValueError("❌ No normal logs (Temp_label=0) found for training.")
            if train_unlabeled_df.count() == 0:
                raise ValueError("❌ No unlabeled logs (Temp_label=999) found for novelty detection.")
            if df_test.count() == 0:
                raise ValueError("❌ No test logs (Temp_label=888) found.")
            if df_val.count() == 0:
                raise ValueError("❌ No val logs (Temp_label=777) found.")

            # ============================================================
            # 2) Deterministic hash key for stable splits
            # ============================================================
            train_normal_df = train_normal_df.withColumn("hid", F.xxhash64(col(id_col)))
            train_unlabeled_df = train_unlabeled_df.withColumn("hid", F.xxhash64(col(id_col)))
            df_test = df_test.withColumn("hid", F.xxhash64(col(id_col)))
            df_val = df_val.withColumn("hid", F.xxhash64(col(id_col)))

            norm_fit_df = train_normal_df.filter((col("hid") % lit(100)) < lit(80))
            norm_holdout_df = train_normal_df.filter((col("hid") % lit(100)) >= lit(80))

            # ============================================================
            # 3) Choose PCA k on NORMAL-fit only
            # ============================================================
            candidate_ks = [10, 20, 40, 50, 60, 70]
            target_variance = 0.999

            best_k = None
            for k in candidate_ks:
                pca_tmp = SparkPCA(k=k, inputCol=feature_col, outputCol=f"pca_k{k}")
                pca_tmp_model = pca_tmp.fit(norm_fit_df)
                explained = float(sum(pca_tmp_model.explainedVariance))
                print(f"[PCA] k={k}, cumulative explained variance={explained:.6f}")
                if explained >= target_variance:
                    best_k = k
                    print(f"[PCA] Selected k={best_k} (reached {target_variance})")
                    break
            if best_k is None:
                best_k = candidate_ks[-1]
                print(f"[PCA] Target variance not reached, using k={best_k}")

            # ============================================================
            # 4) Fit PCA on NORMAL-fit + transform all
            # ============================================================
            pca = SparkPCA(k=best_k, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(norm_fit_df)

            train_pca_normal_fit = pca_model.transform(norm_fit_df)
            train_pca_normal_hold = pca_model.transform(norm_holdout_df)
            train_unlabeled_pca = pca_model.transform(train_unlabeled_df)
            test_pca = pca_model.transform(df_test)
            val_pca = pca_model.transform(df_val)

            # ============================================================
            # 5) PCA reconstruction error
            # ============================================================
            pc = pca_model.pc.toArray()
            d = len(train_pca_normal_fit.select(feature_col).head()[0])
            use_pc_dk = (pc.shape[0] == d)
            pc_b = spark.sparkContext.broadcast(pc)
            use_pc_dk_b = spark.sparkContext.broadcast(use_pc_dk)

            @F.udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray(), dtype=float)
                z = np.array(pca_vec.toArray(), dtype=float)
                pc_local = pc_b.value
                x_hat = (pc_local @ z) if use_pc_dk_b.value else (z @ pc_local)
                return float(np.linalg.norm(x - x_hat))

            train_pca_normal_fit = train_pca_normal_fit.withColumn("anomaly_score_pca",
                                                                   reconstruction_error(col(feature_col),
                                                                                        col("pca_features")))
            train_pca_normal_hold = train_pca_normal_hold.withColumn("anomaly_score_pca",
                                                                     reconstruction_error(col(feature_col),
                                                                                          col("pca_features")))
            train_unlabeled_pca = train_unlabeled_pca.withColumn("anomaly_score_pca",
                                                                 reconstruction_error(col(feature_col),
                                                                                      col("pca_features")))
            test_pca = test_pca.withColumn("anomaly_score_pca",
                                           reconstruction_error(col(feature_col), col("pca_features")))
            val_pca = val_pca.withColumn("anomaly_score_pca",
                                         reconstruction_error(col(feature_col), col("pca_features")))

            # ============================================================
            # 6) PCA threshold (FPR-controlled)
            # ============================================================
            thr_pca = threshold_from_norm_fit(train_pca_normal_fit, "anomaly_score_pca", target_fpr=TARGET_FPR)
            fpr_pca = fpr_on_holdout(train_pca_normal_hold, "anomaly_score_pca", thr_pca)
            print(f"[PCA] thr={thr_pca:.6f}, holdout FPR={fpr_pca:.6f}")

            train_unlabeled_pca = train_unlabeled_pca.withColumn("pseudo_label_pca",
                                                                 when(col("anomaly_score_pca") > lit(thr_pca),
                                                                      1).otherwise(0))

            # ============================================================
            # 7) GMM fit on NORMAL-fit PCA space (choose k by BIC), score by NLL
            # ============================================================
            gmm_ks = [2, 4, 6, 8, 10]
            d_pca = best_k

            def bic_from_model(model, df_features, k, d_pca):
                try:
                    ll = float(model.summary.logLikelihood)
                except Exception:
                    return float("inf")
                n = df_features.count()
                p = (k - 1) + (k * d_pca) + (k * (d_pca * (d_pca + 1) // 2))
                return -2.0 * ll + p * np.log(max(n, 1))

            feat_df = train_pca_normal_fit.select("pca_features")

            best_gmm_model = None
            best_bic = float("inf")
            best_gmm_k = None

            for k in gmm_ks:
                gmm = GaussianMixture(k=k, seed=SEED, featuresCol="pca_features")
                model = gmm.fit(feat_df)
                bic = bic_from_model(model, feat_df, k, d_pca)
                print(f"[GMM] k={k}, BIC={bic}")
                if bic < best_bic:
                    best_bic = bic
                    best_gmm_model = model
                    best_gmm_k = k

            if best_gmm_model is None:
                raise RuntimeError("❌ GMM selection failed.")
            print(f"[GMM] Selected k={best_gmm_k} by BIC.")

            weights = np.array(best_gmm_model.weights, dtype=float)
            gaussians = best_gmm_model.gaussians
            means = np.stack([vec_to_np(g.mean) for g in gaussians], axis=0)
            covs = np.stack([cov_to_2d(g.cov, d_pca) for g in gaussians], axis=0)

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

            @F.udf(DoubleType())
            def gmm_nll(pca_vec):
                z = np.array(pca_vec.toArray(), dtype=float)
                P = bc_params.value
                w = P["weights"]
                m = P["means"]
                ic = P["inv_covs"]
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
                return float(-logp)

            train_gmm_fit = train_pca_normal_fit.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            train_gmm_hold = train_pca_normal_hold.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            unlab_gmm = train_unlabeled_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            test_pca = test_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            val_pca = val_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))

            # ============================================================
            # 8) GMM threshold (FPR-controlled)
            # ============================================================
            thr_gmm = threshold_from_norm_fit(train_gmm_fit, "anomaly_score_gmm", target_fpr=TARGET_FPR)
            fpr_gmm = fpr_on_holdout(train_gmm_hold, "anomaly_score_gmm", thr_gmm)
            print(f"[GMM-NLL] thr={thr_gmm:.6f}, holdout FPR={fpr_gmm:.6f}")

            unlab_gmm = unlab_gmm.withColumn("pseudo_label_gmm",
                                             when(col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0))

            # ============================================================
            # 9) Combine rules
            # ============================================================
            unlab_gmm = (unlab_gmm.withColumn("pseudo_label_and",
                                              when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1),
                                                   1).otherwise(0)).withColumn("pseudo_label_or", when(
                (col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1), 1).otherwise(0)))

            # ============================================================
            # 10) Holdout normal FPR for combined rules
            # ============================================================
            norm_hold_join = (train_gmm_hold.select(id_col, "anomaly_score_gmm").join(
                train_pca_normal_hold.select(id_col, "anomaly_score_pca"), on=id_col, how="inner").withColumn(
                "pca_flag", when(col("anomaly_score_pca") > lit(thr_pca), 1).otherwise(0)).withColumn("gmm_flag", when(
                col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0)).withColumn("and_flag", when(
                (col("pca_flag") == 1) & (col("gmm_flag") == 1), 1).otherwise(0)).withColumn("or_flag", when(
                (col("pca_flag") == 1) | (col("gmm_flag") == 1), 1).otherwise(0)))

            fpr_and = anomaly_rate(norm_hold_join, "and_flag")
            fpr_or = anomaly_rate(norm_hold_join, "or_flag")

            print("\n[HOLDOUT NORMAL FPR]")
            print(f"  PCA-only      : {fpr_pca:.6f}")
            print(f"  GMM-only      : {fpr_gmm:.6f}")
            print(f"  AND           : {fpr_and:.6f}")
            print(f"  OR            : {fpr_or:.6f}")

            # ============================================================
            # 11) Unsupervised selection score (separation * stability / FPR)
            # ============================================================
            SLICE_MOD = 10

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

            u1 = unlab_gmm.filter((col("hid") % lit(100)) < lit(80))
            u2 = unlab_gmm.filter((col("hid") % lit(100)) >= lit(80))

            stab_pca = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_pca")
            stab_gmm = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_gmm")
            stab_and = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_and")
            stab_or = jaccard_anomaly_sets(u1, u2, id_col, "pseudo_label_or")

            score_pca = (sep_pca * stab_pca) / (fpr_pca + eps)
            score_gmm = (sep_gmm * stab_gmm) / (fpr_gmm + eps)
            score_and = (sep_comb * stab_and) / (fpr_and + eps)
            score_or = (sep_comb * stab_or) / (fpr_or + eps)

            print("\n[UNSUP SCORE]")
            print(f"  PCA : sep={sep_pca:.3f} stab={stab_pca:.3f} fpr={fpr_pca:.5f} score={score_pca:.6f}")
            print(f"  GMM : sep={sep_gmm:.3f} stab={stab_gmm:.3f} fpr={fpr_gmm:.5f} score={score_gmm:.6f}")
            print(f"  AND : sep={sep_comb:.3f} stab={stab_and:.3f} fpr={fpr_and:.5f} score={score_and:.6f}")
            print(f"  OR  : sep={sep_comb:.3f} stab={stab_or:.3f} fpr={fpr_or:.5f} score={score_or:.6f}")

            best_method = \
            max([("pca", score_pca), ("gmm", score_gmm), ("and", score_and), ("or", score_or)], key=lambda x: x[1])[0]
            if METHOD.lower() in ["pca", "gmm", "and", "or"]:
                best_method = METHOD.lower()

            print(f"\n[SELECTED NOVELTY METHOD] {best_method}")

            if best_method == "pca":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_pca"))
            elif best_method == "gmm":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_gmm"))
            elif best_method == "and":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_and"))
            else:
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_or"))

            # ============================================================
            # 12) CONFIDENCE FILTERING + optional anomaly cap
            # ============================================================
            train_norm_scores = (
                train_gmm_fit.select(id_col, feature_col, "anomaly_score_pca", "anomaly_score_gmm").withColumn(
                    "comb_score", col("anomaly_score_pca") + col("anomaly_score_gmm")))

            unlab_scores = (unlab_gmm.select(id_col, feature_col, "Final_Label", "anomaly_score_pca",
                                             "anomaly_score_gmm").withColumn("comb_score",
                                                                             col("anomaly_score_pca") + col(
                                                                                 "anomaly_score_gmm")))

            thr_comb = threshold_from_norm_fit(train_norm_scores, "comb_score", target_fpr=TARGET_FPR)

            conf_anom = unlab_scores.filter(
                (col("Final_Label") == 1) & (col("anomaly_score_pca") > lit(thr_pca * (1.0 + MARGIN_PCA))) & (
                            col("anomaly_score_gmm") > lit(thr_gmm * (1.0 + MARGIN_GMM))) & (
                            col("comb_score") > lit(thr_comb * (1.0 + MARGIN_COMB))))

            conf_norm = unlab_scores.filter(
                (col("Final_Label") == 0) & (col("anomaly_score_pca") < lit(thr_pca * (1.0 - MARGIN_PCA))) & (
                            col("anomaly_score_gmm") < lit(thr_gmm * (1.0 - MARGIN_GMM))) & (
                            col("comb_score") < lit(thr_comb * (1.0 - MARGIN_COMB))))

            if CAP_PSEUDO_ANOM_TOP_PCT is not None:
                # keep only top X% anomalies by comb_score (reduces anomaly bias)
                anom = conf_anom
                if anom.count() > 0:
                    thr_top = approx_quantile(anom, "comb_score", 1.0 - float(CAP_PSEUDO_ANOM_TOP_PCT), rel=1e-3)
                    conf_anom = anom.filter(col("comb_score") >= lit(thr_top))
                    print(
                        f"[CAP] Keeping top {CAP_PSEUDO_ANOM_TOP_PCT * 100:.1f}% pseudo anomalies by comb_score. thr_top={thr_top:.6f}")

            train_df_normal_cls = (
                norm_fit_df.select(id_col, feature_col).withColumn("train_used_label", lit(0).cast("int")))

            train_df_unlabeled_cls = (conf_anom.unionByName(conf_norm).select(id_col, feature_col,
                                                                              col("Final_Label").cast("int").alias(
                                                                                  "train_used_label")))

            df_final_train_cls = train_df_normal_cls.unionByName(train_df_unlabeled_cls, allowMissingColumns=False)

            print("\n[FINAL TRAIN BALANCE] after filtering/cap:")
            df_final_train_cls.groupBy("train_used_label").count().orderBy("train_used_label").show()
            print("[UNLABELED] total:", train_unlabeled_df.count())
            print("[UNLABELED] kept :", train_df_unlabeled_cls.count())

            # ============================================================
            # 13) Build val/test for supervised model
            # ============================================================
            df_val_cls = df_val.select(id_col, feature_col, col("Label").cast("int").alias("label"))
            df_test_cls = df_test.select(id_col, feature_col, col("Label").cast("int").alias("label"))

            # ============================================================
            # 14) REPORT 1: UNLABELED pseudo-label quality (if true labels exist)
            # ============================================================
            try:
                unlabeled_eval = (unlab_gmm.select(id_col, col("Final_Label").cast("int").alias("pseudo_label")).join(
                    sequences_df.select(id_col, col("Label").cast("int").alias("true_label")), on=id_col,
                    how="inner").dropna())
                pdf_u = to_pandas_report(unlabeled_eval, "true_label", "pseudo_label")
                if pdf_u is not None:
                    print("\n=== REPORT 1: UNLABELED (true_label vs pseudo_label) ===")
                    print(classification_report(pdf_u["true_label"], pdf_u["pseudo_label"], digits=3))
                else:
                    print("\n[WARN] REPORT 1 skipped: no rows.")
            except Exception as e:
                print(f"\n[WARN] REPORT 1 skipped: {e}")

            # ============================================================
            # 15) REPORT 2: FINAL TRAIN SET label quality (if true labels exist)
            # ============================================================
            try:
                train_quality = (
                    df_final_train_cls.join(sequences_df.select(id_col, col("Label").cast("int").alias("true_label")),
                                            on=id_col, how="inner").select("true_label", "train_used_label").dropna())
                pdf_tr = to_pandas_report(train_quality, "true_label", "train_used_label")
                if pdf_tr is not None:
                    print("\n=== REPORT 2: FINAL TRAIN SET (true_label vs train_used_label) ===")
                    print(classification_report(pdf_tr["true_label"], pdf_tr["train_used_label"], digits=3))
                else:
                    print("\n[WARN] REPORT 2 skipped: no rows.")
            except Exception as e:
                print(f"\n[WARN] REPORT 2 skipped: {e}")

            # ============================================================
            # 16) Train SUPERVISED MODEL (GBT)
            # ============================================================
            train_for_ml = (
                df_final_train_cls.withColumnRenamed("train_used_label", "label").select(feature_col, "label"))

            gbt = GBTClassifier(featuresCol=feature_col, labelCol="label", maxIter=GBT_MAX_ITER, maxDepth=GBT_MAX_DEPTH,
                stepSize=GBT_STEP_SIZE, seed=SEED)

            gbt_model = gbt.fit(train_for_ml)
            print("\n[INFO] Supervised model trained: GBTClassifier")

            # ============================================================
            # 17) Tune threshold on VAL (maximize F1)
            # ============================================================
            best_thr = tune_threshold_on_val(gbt_model, df_val_cls, features_col=feature_col, label_col="label")
            print(f"\n[THRESHOLD] Best threshold on VAL (max F1): {best_thr:.6f}")

            # show VAL report at tuned threshold
            pred_val = gbt_model.transform(df_val_cls).select(col("label").cast("int").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("p1"))
            pdf_val = pred_val.toPandas()
            y_val = pdf_val["y"].astype(int).values
            p_val = pdf_val["p1"].astype(float).values
            print("\n=== VAL report @ tuned threshold ===")
            print(classification_report(y_val, (p_val >= best_thr).astype(int), digits=3))

            # ============================================================
            # 18) REPORT 3: TEST (true label vs supervised prediction @ tuned threshold)
            # ============================================================
            pred_test = gbt_model.transform(df_test_cls).select(col("label").cast("int").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("p1"), col("rawPrediction"))

            pdf_test = pred_test.toPandas()
            y_te = pdf_test["y"].astype(int).values
            p_te = pdf_test["p1"].astype(float).values

            print("\n=== REPORT 3: TEST (true label vs supervised prediction @ tuned threshold) ===")
            print(classification_report(y_te, (p_te >= best_thr).astype(int), digits=3))

            # Optional: TEST AUC
            evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                                      metricName="areaUnderROC")
            auc_test = evaluator.evaluate(gbt_model.transform(df_test_cls))
            print(f"[TEST] AUC = {auc_test:.4f}")
            '''

            #----------------------- (2)
            sequences_df.printSchema()
            sequences_df.groupBy("Label").count().show()
            sequences_df.groupBy("Temp_label").count().show()

            # --- Robust label normalization (fix casing/whitespace) ---
            sequences_df = sequences_df.withColumn("Label_str", F.lower(F.trim(F.col("Label").cast("string"))))

            sequences_df = sequences_df.withColumn("Label",
                F.when(F.col("Label_str").isin("normal", "0"), F.lit(0)).when(F.col("Label_str").isin("anomaly", "1"),
                                                                              F.lit(1)).otherwise(
                    F.lit(None).cast("int"))).drop("Label_str")

            # sanity check: fail fast if unknown values exist
            bad = sequences_df.filter(F.col("Label").isNull()).select("Label").count()
            if bad > 0:
                sequences_df.filter(F.col("Label").isNull()).select("Label").show(50, False)
                raise ValueError("Unknown label values after normalization.")


            print(f"\n🚀 Starting novelty detection using method = {method.upper()}")

            # ---------------------------------------------------------
            # 1️⃣ Split data by Temp_label
            # ---------------------------------------------------------
            train_normal_df = sequences_df.filter(col("Temp_label") == 0)
            train_unlabeled_df = sequences_df.filter(col("Temp_label") == 999)
            df_test = sequences_df.filter(col("Temp_label") == 888)
            df_val = sequences_df.filter(col("Temp_label") == 777)

            print('info label  ..........................................')
            train_normal_df.groupBy("Label").count().show()
            train_unlabeled_df.groupBy("Label").count().show()
            df_test.groupBy("Label").count().show()
            df_val.groupBy("Label").count().show()

            if train_normal_df.count() == 0:
                raise ValueError("❌ No normal logs (Temp_label=0) found for training.")
            if train_unlabeled_df.count() == 0:
                raise ValueError("❌ No unlabeled logs (Temp_label=999) found for novelty detection.")

            # ---------------------------------------------------------
            # 2️⃣ Settings / columns
            # ---------------------------------------------------------
            feature_col = "features_vec_final"
            id_col = "Node_block_id"

            TARGET_FPR = 0.01
            eps = 1e-9

            # ---------------------------------------------------------
            # 3️⃣ Deterministic key + deterministic normal split (80/20)
            # ---------------------------------------------------------
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

            print("\n🧠 Using PCA + GMM for novelty detection (updated/stable) ...")

            # ============================================================
            # 4) Choose PCA k on NORMAL-fit only
            # ============================================================
            candidate_ks = [10, 20, 40, 50, 60, 70]
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
            # 5) Fit PCA on NORMAL-fit + transform all
            # ============================================================
            pca = SparkPCA(k=best_k, inputCol=feature_col, outputCol="pca_features")
            pca_model = pca.fit(norm_fit_df)

            train_pca_normal_fit = pca_model.transform(norm_fit_df)
            train_pca_normal_hold = pca_model.transform(norm_holdout_df)

            train_unlabeled_pca = pca_model.transform(train_unlabeled_df)
            test_pca = pca_model.transform(df_test)
            val_pca = pca_model.transform(df_val)

            # ============================================================
            # 6) PCA reconstruction error
            # ============================================================
            pc = pca_model.pc.toArray()
            d = len(train_pca_normal_fit.select(feature_col).head()[0])
            use_pc_dk = (pc.shape[0] == d)

            pc_b = spark.sparkContext.broadcast(pc)
            use_pc_dk_b = spark.sparkContext.broadcast(use_pc_dk)

            @F.udf(DoubleType())
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
            test_pca = test_pca.withColumn("anomaly_score_pca",
                reconstruction_error(col(feature_col), col("pca_features")))
            val_pca = val_pca.withColumn("anomaly_score_pca",
                reconstruction_error(col(feature_col), col("pca_features")))

            # ============================================================
            # 7) PCA threshold (FPR-controlled)
            # ============================================================
            thr_pca = threshold_from_norm_fit(train_pca_normal_fit, "anomaly_score_pca", target_fpr=TARGET_FPR)
            fpr_pca = fpr_on_holdout(train_pca_normal_hold, "anomaly_score_pca", thr_pca)
            print(f"[PCA] thr={thr_pca:.6f}, holdout FPR={fpr_pca:.6f}")

            train_unlabeled_pca = train_unlabeled_pca.withColumn("pseudo_label_pca",
                when(col("anomaly_score_pca") > lit(thr_pca), 1).otherwise(0))

            # ============================================================
            # 8) Fit GMM on NORMAL-fit PCA space (choose k by BIC), score by NLL
            # ============================================================
            gmm_ks = [2, 4, 6, 8, 10]
            SEED = 123
            d_pca = best_k

            def bic_from_model(model, df_features, k, d_pca):
                try:
                    ll = float(model.summary.logLikelihood)
                except Exception:
                    return float("inf")
                n = df_features.count()
                p = (k - 1) + (k * d_pca) + (k * (d_pca * (d_pca + 1) // 2))
                return -2.0 * ll + p * np.log(max(n, 1))

            feat_df = train_pca_normal_fit.select("pca_features")

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

            # ---- Robust extraction of means/covs ----
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

            means = np.stack([vec_to_np(g.mean) for g in gaussians], axis=0)
            covs = np.stack([cov_to_2d(g.cov, d_pca) for g in gaussians], axis=0)

            # jitter
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

            @F.udf(DoubleType())
            def gmm_nll(pca_vec):
                z = np.array(pca_vec.toArray(), dtype=float)
                P = bc_params.value
                w = P["weights"]
                m = P["means"]
                ic = P["inv_covs"]
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
                return float(-logp)

            train_gmm_fit = train_pca_normal_fit.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            train_gmm_hold = train_pca_normal_hold.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            unlab_gmm = train_unlabeled_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))

            test_pca = test_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            val_pca = val_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))

            # ============================================================
            # 9) GMM threshold (FPR-controlled) + pseudo labels
            # ============================================================
            thr_gmm = threshold_from_norm_fit(train_gmm_fit, "anomaly_score_gmm", target_fpr=TARGET_FPR)
            fpr_gmm = fpr_on_holdout(train_gmm_hold, "anomaly_score_gmm", thr_gmm)
            print(f"[GMM-NLL] thr={thr_gmm:.6f}, holdout FPR={fpr_gmm:.6f}")

            unlab_gmm = unlab_gmm.withColumn("pseudo_label_gmm",
                when(col("anomaly_score_gmm") > lit(thr_gmm), 1).otherwise(0))

            # ============================================================
            # 10) Combine rules (AND / OR)
            # ============================================================
            unlab_gmm = unlab_gmm.withColumn("pseudo_label_and",
                when((col("pseudo_label_pca") == 1) & (col("pseudo_label_gmm") == 1), 1).otherwise(0)).withColumn(
                "pseudo_label_or",
                when((col("pseudo_label_pca") == 1) | (col("pseudo_label_gmm") == 1), 1).otherwise(0))

            # ============================================================
            # 10.1) Holdout normal FPR for combined rules
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
            # 10.2) Unsupervised selection (fixed) + stability (FIXED u1/u2)
            # ============================================================
            SLICE_MOD = 10  # deterministic ~10% slice for pandas
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

            # FIX: true complementary split (80/20), not overlapping
            u1 = unlab_gmm.filter((col("hid") % lit(100)) < lit(80))
            u2 = unlab_gmm.filter((col("hid") % lit(100)) >= lit(80))

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

            # user override (optional)
            if method.lower() in ["pca", "gmm", "and", "or"]:
                best_method = method.lower()

            print(f"\n[SELECTED] Best method: {best_method}")

            if best_method == "pca":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_pca"))
            elif best_method == "gmm":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_gmm"))
            elif best_method == "and":
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_and"))
            else:
                unlab_gmm = unlab_gmm.withColumn("Final_Label", col("pseudo_label_or"))

            # ============================================================
            # 11) DEBUG: evaluation on "unlabeled" only if true labels exist
            # ============================================================
            try:
                train_unlabeled_eval_df = unlab_gmm.join(
                    sequences_df.select(col(id_col), col("Label").alias("true_label")), on=id_col, how="inner")
                pdf_unlabeled = train_unlabeled_eval_df.select("true_label", "pseudo_label_pca", "pseudo_label_gmm",
                    "pseudo_label_and", "pseudo_label_or", "Final_Label").toPandas()

                print("\n=== Classification_report on unlabeled (SELECTED Final_Label) ===")
                print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["Final_Label"], digits=3))
            except Exception as e:
                print(f"[DEBUG] Skipping debug evaluation: {e}")

            # ============================================================
            # 12) Build final training set WITHOUT confidence filtering (Option A)
            # ============================================================
            keep_cols = [id_col, "pca_features", "Final_Label"]

            # True normal part (label=0)
            train_df_normal_cls = (
                train_pca_normal_fit.withColumn("Final_Label", lit(0).cast("int")).select(*keep_cols))

            # Keep ALL pseudo-labeled unlabeled rows (no filtering)
            train_df_unlabeled_cls = (
                unlab_gmm.withColumn("Final_Label", col("Final_Label").cast("int")).select(*keep_cols))

            df_final_train_cls = train_df_normal_cls.unionByName(train_df_unlabeled_cls, allowMissingColumns=False)

            print("\n[CHECK] Final train class balance (NO confidence filtering):")
            # df_final_train_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()
            # print("[DEBUG] unlabeled total:", train_unlabeled_df.count())
            # print("[DEBUG] unlabeled used (no filter):", train_df_unlabeled_cls.count())

            '''
            # ============================================================
            # 12) Build final training set with CONFIDENCE FILTERING (IMPORTANT)
            # ============================================================
            keep_cols = [id_col, "pca_features", "Final_Label"]

            # True normal part (label=0)
            train_df_normal_cls = (
                train_pca_normal_fit.withColumn("Final_Label", lit(0).cast("int")).select(*keep_cols))

            # Confidence filtering on pseudo labels using distance to thresholds
            # Keep only confident pseudo anomalies + confident pseudo normals
            # Tune margins if needed
            MARGIN_PCA = 0.15
            MARGIN_GMM = 0.15

            # Define a combined confidence score (higher => more anomalous)
            unlab_gmm = unlab_gmm.withColumn("comb_score", col("anomaly_score_pca") + col("anomaly_score_gmm"))

            # create conservative thresholds on combined score based on normal-fit distribution
            train_gmm_fit = train_gmm_fit.withColumn("comb_score", col("anomaly_score_pca") + col("anomaly_score_gmm"))
            thr_comb = threshold_from_norm_fit(train_gmm_fit, "comb_score", target_fpr=TARGET_FPR)

            # confident anomalies: far above threshold(s)
            conf_anom = unlab_gmm.filter(
                (col("Final_Label") == 1) & (col("anomaly_score_pca") > lit(thr_pca * (1.0 + MARGIN_PCA))) & (
                            col("anomaly_score_gmm") > lit(thr_gmm * (1.0 + MARGIN_GMM))) & (
                            col("comb_score") > lit(thr_comb * (1.0 + 0.10))))

            # confident normals: far below threshold(s)
            conf_norm = unlab_gmm.filter(
                (col("Final_Label") == 0) & (col("anomaly_score_pca") < lit(thr_pca * (1.0 - MARGIN_PCA))) & (
                            col("anomaly_score_gmm") < lit(thr_gmm * (1.0 - MARGIN_GMM))) & (
                            col("comb_score") < lit(thr_comb * (1.0 - 0.10))))

            train_df_unlabeled_cls = (
                conf_anom.unionByName(conf_norm).withColumn("Final_Label", col("Final_Label").cast("int")).select(
                    *keep_cols))

            df_final_train_cls = train_df_normal_cls.unionByName(train_df_unlabeled_cls, allowMissingColumns=False)

            print("\n[CHECK] Final train class balance (after confidence filtering):")
            df_final_train_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()
            '''


            # ============================================================
            # 13) Prepare test/val (FIX: check Label in *test_pca/val_pca*)
            # ============================================================
            test_has_label = "Label" in test_pca.columns
            val_has_label = "Label" in val_pca.columns

            df_test_cls = (test_pca.withColumn("Final_Label",
                (col("Label").cast("int") if test_has_label else lit(None).cast("int"))).select(id_col, "pca_features",
                                                                                                "Final_Label"))

            df_val_cls = (val_pca.withColumn("Final_Label",
                (col("Label").cast("int") if val_has_label else lit(None).cast("int"))).select(id_col, "pca_features",
                                                                                               "Final_Label"))

            # Sanity: no null labels in val/test if Label exists in sequences_df
            if test_has_label:
                n_null = df_test_cls.filter(col("Final_Label").isNull()).count()
                if n_null > 0:
                    raise ValueError(f"[ERROR] test Final_Label has {n_null} NULLs (Label existed but got lost).")
            if val_has_label:
                n_null = df_val_cls.filter(col("Final_Label").isNull()).count()
                if n_null > 0:
                    raise ValueError(f"[ERROR] val Final_Label has {n_null} NULLs (Label existed but got lost).")

            print("\n[CHECK] Val/Test class balance:")
            df_val_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()
            df_test_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()

            # ============================================================
            # 14) FINAL TRAIN QUALITY REPORT (true Label vs Final_Label)
            # ============================================================
            if "Label" in sequences_df.columns:
                df_train_quality = (
                    df_final_train_cls.join(sequences_df.select(col(id_col), col("Label").alias("true_label")),
                                            on=id_col, how="inner")) # .select("Node_block_id", "pca_features", "true_label", "Final_Label").dropna()
                n_quality = df_train_quality.count()
                if n_quality > 0:
                    pdf_train_quality = df_train_quality.toPandas()
                    pdf_train_quality["true_label"] = pdf_train_quality["true_label"].astype(int)
                    pdf_train_quality["Final_Label"] = pdf_train_quality["Final_Label"].astype(int)

                    print("\n=== Classification_report on unlabeled (SELECTED Final_Label) ===")
                    print(classification_report(pdf_unlabeled["true_label"], pdf_unlabeled["Final_Label"], digits=3))

                    print(f"\n=== Nayef: Classification_report on FINAL TRAIN SET (n={n_quality}) ===")
                    print(classification_report(pdf_train_quality["true_label"], pdf_train_quality["Final_Label"],
                                                digits=3))
                else:
                    print("[WARN] No rows with both true_label and Final_Label -> report skipped.")
            else:
                print(
                    "[WARN] sequences_df has no 'Label' column -> cannot compute final training classification report.")
            #exit()


            print("=== SCHEMA ===")
            #df_final_train_cls.printSchema()
            #df_val_cls.printSchema()
            #df_test_cls.printSchema()

            #print("=== TRAIN Final_Label distribution ===")
            #df_final_train_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()

            #print("=== VAL Final_Label distribution ===")
            #df_val_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()

            #print("=== TEST Final_Label distribution ===")
            #df_test_cls.groupBy("Final_Label").count().orderBy("Final_Label").show()

            # ============================================================
            # SUPERVISED CLASSIFIER TRAINING AFTER NOVELTY/PSEUDO-LABELING
            # Copy-paste ready (Spark ML)
            # Assumes you already created:
            #   df_final_train_cls  (id_col, pca_features, Final_Label)
            #   df_val_cls          (id_col, pca_features, Final_Label)
            #   df_test_cls         (id_col, pca_features, Final_Label)
            # ============================================================

            from pyspark.sql import functions as F
            from pyspark.sql.functions import col, lit, when
            from pyspark.ml import Pipeline
            from pyspark.ml.classification import LogisticRegression
            from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

            '''

            # -----------------------------
            # Settings
            # -----------------------------
            id_col = "Node_block_id"
            feat_col = "pca_features"
            label_col = "Final_Label"

            TARGET_FPR = 0.01
            REL_ERR_Q = 1e-3
            EPS = 1e-9

            # -----------------------------
            # Prepare supervised datasets
            # -----------------------------
            def prep_sup(df, name):
                out = (df.select(col(id_col), col(feat_col), col(label_col).cast("int").alias(label_col)).dropna(
                    subset=[feat_col, label_col]))
                print(f"\n[{name}] class distribution:")
                out.groupBy(label_col).count().orderBy(label_col).show()
                return out

            train_sup = prep_sup(df_final_train_cls, "TRAIN_SUP")
            val_sup = prep_sup(df_val_cls, "VAL_SUP")
            test_sup = prep_sup(df_test_cls, "TEST_SUP")

            # -----------------------------
            # Class weights (used by LR; for trees we also keep it available)
            # -----------------------------
            counts = train_sup.groupBy(label_col).count().collect()
            cnt = {int(r[label_col]): int(r["count"]) for r in counts}

            n0 = float(cnt.get(0, 1))
            n1 = float(cnt.get(1, 1))
            total = n0 + n1
            w0 = total / (2.0 * n0) if n0 > 0 else 1.0
            w1 = total / (2.0 * n1) if n1 > 0 else 1.0

            train_sup_w = train_sup.withColumn("classWeightCol", when(col(label_col) == 1, lit(w1)).otherwise(lit(w0)))

            # -----------------------------
            # Helpers
            # -----------------------------
            auc_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")

            def add_p1(df_pred):
                return df_pred.withColumn("p1", vector_to_array(col("probability"))[1].cast("double"))

            def apply_threshold_pred(scored_df, thr_value):
                return scored_df.withColumn("pred_thr", when(col("p1") >= lit(thr_value), lit(1)).otherwise(lit(0)))

            def print_class_metrics(pred_df, pred_col, title):
                tp = pred_df.filter((col(label_col) == 1) & (col(pred_col) == 1)).count()
                tn = pred_df.filter((col(label_col) == 0) & (col(pred_col) == 0)).count()
                fp = pred_df.filter((col(label_col) == 0) & (col(pred_col) == 1)).count()
                fn = pred_df.filter((col(label_col) == 1) & (col(pred_col) == 0)).count()

                precision_1 = tp / (tp + fp + EPS)
                recall_1 = tp / (tp + fn + EPS)
                f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1 + EPS)

                precision_0 = tn / (tn + fn + EPS)
                recall_0 = tn / (tn + fp + EPS)
                f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0 + EPS)

                fpr = fp / (fp + tn + EPS)
                acc = (tp + tn) / (tp + tn + fp + fn + EPS)

                print("\n===================================================")
                print(title)
                print("Confusion Matrix:")
                print(f"TP={tp}  FP={fp}")
                print(f"FN={fn}  TN={tn}")
                print(f"ACC={acc:.6f}  FPR={fpr:.6f}")

                print("\nClass 1 (Anomaly)")
                print(f"Precision: {precision_1:.6f}")
                print(f"Recall   : {recall_1:.6f}")
                print(f"F1-score : {f1_1:.6f}")

                print("\nClass 0 (Normal)")
                print(f"Precision: {precision_0:.6f}")
                print(f"Recall   : {recall_0:.6f}")
                print(f"F1-score : {f1_0:.6f}")
                print("===================================================\n")

            def train_and_eval(model, model_name, use_weights_for_fit):
                print(f"\n\n==================== {model_name} ====================")

                fit_df = train_sup_w if use_weights_for_fit else train_sup

                pipe = Pipeline(stages=[model])
                fitted = pipe.fit(fit_df)

                # --- default threshold eval (AUC only) ---
                val_pred = fitted.transform(val_sup)
                test_pred = fitted.transform(test_sup)

                val_auc = auc_eval.evaluate(val_pred)
                test_auc = auc_eval.evaluate(test_pred)
                print(f"[AUC] VAL={val_auc:.6f}  TEST={test_auc:.6f}")

                # --- tune threshold on VAL normals to hit TARGET_FPR ---
                val_scored = add_p1(val_pred).cache()
                thr = float(
                    val_scored.filter(col(label_col) == 0).approxQuantile("p1", [1.0 - TARGET_FPR], REL_ERR_Q)[0])
                print(f"[THR tuned @ VAL normals FPR~{TARGET_FPR}] thr={thr:.6f}")

                # --- apply threshold + metrics ---
                val_thr_pred = apply_threshold_pred(val_scored, thr)
                test_scored = add_p1(test_pred)
                test_thr_pred = apply_threshold_pred(test_scored, thr)

                print_class_metrics(val_thr_pred, "pred_thr", f"{model_name} - VAL (tuned thr={thr:.6f})")
                print_class_metrics(test_thr_pred, "pred_thr", f"{model_name} - TEST (tuned thr={thr:.6f})")

                return fitted, thr

            # ============================================================
            # Models to try
            # ============================================================

            # 1) Logistic Regression (baseline)
            lr = LogisticRegression(featuresCol=feat_col, labelCol=label_col, weightCol="classWeightCol", maxIter=200,
                regParam=0.01, elasticNetParam=0.0)

            # 2) Random Forest (often improves precision)
            rf = RandomForestClassifier(featuresCol=feat_col, labelCol=label_col, numTrees=300, maxDepth=10,
                featureSubsetStrategy="sqrt", seed=123)
            # Note: many Spark versions do NOT support weightCol for RF. If yours does, you can add weightCol="classWeightCol"

            # 3) Gradient-Boosted Trees (often best)
            gbt = GBTClassifier(featuresCol=feat_col, labelCol=label_col, maxIter=200, maxDepth=5, stepSize=0.05,
                subsamplingRate=0.8, seed=123)
            # Note: many Spark versions do NOT support weightCol for GBT either.

            # ============================================================
            # Run all
            # ============================================================
            #lr_model, lr_thr = train_and_eval(lr, "LogisticRegression", use_weights_for_fit=True)
            rf_model, rf_thr = train_and_eval(rf, "RandomForest", use_weights_for_fit=False)
            #gbt_model, gbt_thr = train_and_eval(gbt, "GBTClassifier", use_weights_for_fit=False)

            print("\nDone. Choose the model with best TEST class-1 F1/precision/recall under tuned threshold.")
            exit()
            '''
            '''
            # -----------------------------
            # Settings
            # -----------------------------
            id_col = "Node_block_id"
            feat_col = "pca_features"
            label_col = "Final_Label"

            TARGET_FPR = 0.01  # target false positive rate on VAL normals
            REL_ERR_Q = 1e-3  # approxQuantile relative error
            EPS = 1e-9

            # ============================================================
            # 1) Prepare supervised datasets
            # ============================================================
            def prep_sup(df, name):
                out = (df.select(col(id_col), col(feat_col), col(label_col).cast("int").alias(label_col)).dropna(
                    subset=[feat_col, label_col]))
                print(f"\n[{name}] class distribution:")
                out.groupBy(label_col).count().orderBy(label_col).show()
                return out

            train_sup = prep_sup(df_final_train_cls, "TRAIN_SUP")
            val_sup = prep_sup(df_val_cls, "VAL_SUP")
            test_sup = prep_sup(df_test_cls, "TEST_SUP")

            # ============================================================
            # 2) Add class weights (handle imbalance)
            # ============================================================
            counts = train_sup.groupBy(label_col).count().collect()
            cnt = {int(r[label_col]): int(r["count"]) for r in counts}

            n0 = float(cnt.get(0, 1))
            n1 = float(cnt.get(1, 1))
            total = n0 + n1

            w0 = total / (2.0 * n0) if n0 > 0 else 1.0
            w1 = total / (2.0 * n1) if n1 > 0 else 1.0

            print(f"\n[WEIGHTS] n0={n0:.0f}, n1={n1:.0f}, w0={w0:.6f}, w1={w1:.6f}")

            train_sup_w = train_sup.withColumn("classWeightCol", when(col(label_col) == 1, lit(w1)).otherwise(lit(w0)))

            # ============================================================
            # 3) Train classifier (Logistic Regression)
            # ============================================================
            lr = LogisticRegression(featuresCol=feat_col, labelCol=label_col, weightCol="classWeightCol", maxIter=200,
                regParam=0.01, elasticNetParam=0.0)

            pipe = Pipeline(stages=[lr])
            clf_model = pipe.fit(train_sup_w)
            print("\n[MODEL] Trained LogisticRegression.")

            # ============================================================
            # 4) Helpers for probability + metrics
            # ============================================================
            auc_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction",
                metricName="areaUnderROC")
            acc_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction",
                metricName="accuracy")
            f1_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")

            def add_p1(df_pred):
                # probability is VectorUDT -> convert to array -> take index 1
                return df_pred.withColumn("p1", vector_to_array(col("probability"))[1].cast("double"))

            def print_class_metrics(pred_df, pred_col, title):
                tp = pred_df.filter((col(label_col) == 1) & (col(pred_col) == 1)).count()
                tn = pred_df.filter((col(label_col) == 0) & (col(pred_col) == 0)).count()
                fp = pred_df.filter((col(label_col) == 0) & (col(pred_col) == 1)).count()
                fn = pred_df.filter((col(label_col) == 1) & (col(pred_col) == 0)).count()

                precision_1 = tp / (tp + fp + EPS)
                recall_1 = tp / (tp + fn + EPS)
                f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1 + EPS)

                precision_0 = tn / (tn + fn + EPS)
                recall_0 = tn / (tn + fp + EPS)
                f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0 + EPS)

                acc = (tp + tn) / (tp + tn + fp + fn + EPS)
                fpr = fp / (fp + tn + EPS)

                print("\n===================================================")
                print(title)
                print("Confusion Matrix:")
                print(f"TP={tp}  FP={fp}")
                print(f"FN={fn}  TN={tn}")

                print("\nClass 1 (Anomaly)")
                print(f"Precision: {precision_1:.6f}")
                print(f"Recall   : {recall_1:.6f}")
                print(f"F1-score : {f1_1:.6f}")

                print("\nClass 0 (Normal)")
                print(f"Precision: {precision_0:.6f}")
                print(f"Recall   : {recall_0:.6f}")
                print(f"F1-score : {f1_0:.6f}")

                print(f"\nOverall Accuracy: {acc:.6f}")
                print(f"FPR (Normal->Anomaly): {fpr:.6f}")
                print("===================================================\n")

            def evaluate_default(df, name):
                pred = clf_model.transform(df)
                auc = auc_eval.evaluate(pred)
                acc = acc_eval.evaluate(pred)
                f1w = f1_eval.evaluate(pred)

                print(f"\n[{name}] DEFAULT threshold (Spark prediction)")
                print(f"AUC={auc:.6f}  ACC={acc:.6f}  F1(weighted)={f1w:.6f}")

                print_class_metrics(pred, "prediction", f"{name} - DEFAULT threshold")
                return pred

            def apply_threshold(df, thr_value):
                scored = add_p1(clf_model.transform(df))
                return scored.withColumn("pred_thr", when(col("p1") >= lit(thr_value), lit(1)).otherwise(lit(0)))

            def evaluate_custom(df, name, thr_value):
                pred = apply_threshold(df, thr_value)
                print_class_metrics(pred, "pred_thr", f"{name} - CUSTOM threshold @ thr={thr_value:.6f}")
                return pred

            # ============================================================
            # 5) Evaluate DEFAULT threshold
            # ============================================================
            val_pred_default = evaluate_default(val_sup, "VAL")
            test_pred_default = evaluate_default(test_sup, "TEST")

            # ============================================================
            # 6) Tune threshold on VAL normals for TARGET_FPR
            # ============================================================
            val_scored = add_p1(clf_model.transform(val_sup)).cache()
            val_normals = val_scored.filter(col(label_col) == 0)

            thr = float(val_normals.approxQuantile("p1", [1.0 - TARGET_FPR], REL_ERR_Q)[0])
            print(f"\n[THRESHOLD] Selected thr={thr:.6f} to target VAL normal FPR~{TARGET_FPR:.4f}")

            # ============================================================
            # 7) Evaluate CUSTOM threshold
            # ============================================================
            val_pred_thr = evaluate_custom(val_sup, "VAL", thr)
            test_pred_thr = evaluate_custom(test_sup, "TEST", thr)

            # ============================================================
            # 8) Output predictions on TEST (id + prob + pred + true)
            # ============================================================
            test_out = (apply_threshold(test_sup, thr).select(col(id_col), col("p1").alias("prob_anomaly"),
                col("pred_thr").alias("pred_label"), col(label_col).alias("true_label")))

            print("\n[TEST OUTPUT SAMPLE]")
            test_out.show(20, False)
            '''





            #--------xx
            # ---------------------------------------
            # SETTINGS
            # ---------------------------------------
            label_col = "Final_Label"

            # Use ONE of these:
            # pred_df = test_pred_default      # if using default Spark threshold
            # pred_col = "prediction"

            pred_df = test_pred_thr  # if using your custom threshold
            pred_col = "pred_thr"

            # ---------------------------------------
            # Compute Confusion Matrix Counts
            # ---------------------------------------
            tp = pred_df.filter((col(label_col) == 1) & (col(pred_col) == 1)).count()
            tn = pred_df.filter((col(label_col) == 0) & (col(pred_col) == 0)).count()
            fp = pred_df.filter((col(label_col) == 0) & (col(pred_col) == 1)).count()
            fn = pred_df.filter((col(label_col) == 1) & (col(pred_col) == 0)).count()

            eps = 1e-9

            # ---------------------------------------
            # Class 1 metrics
            # ---------------------------------------
            precision_1 = tp / (tp + fp + eps)
            recall_1 = tp / (tp + fn + eps)
            f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1 + eps)

            # ---------------------------------------
            # Class 0 metrics
            # ---------------------------------------
            precision_0 = tn / (tn + fn + eps)
            recall_0 = tn / (tn + fp + eps)
            f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0 + eps)

            # ---------------------------------------
            # Print Results
            # ---------------------------------------
            print("\n================ CLASSIFICATION REPORT =================")
            print("Confusion Matrix:")
            print(f"TP={tp}  FP={fp}")
            print(f"FN={fn}  TN={tn}")

            print("\nClass 1 (Anomaly)")
            print(f"Precision: {precision_1:.6f}")
            print(f"Recall   : {recall_1:.6f}")
            print(f"F1-score : {f1_1:.6f}")

            print("\nClass 0 (Normal)")
            print(f"Precision: {precision_0:.6f}")
            print(f"Recall   : {recall_0:.6f}")
            print(f"F1-score : {f1_0:.6f}")

            # ---------------------------------------
            # Optional: Overall accuracy
            # ---------------------------------------
            accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
            print(f"\nOverall Accuracy: {accuracy:.6f}")
            print("========================================================\n")

            return df_final_train_cls, df_test_cls, df_val_cls
            #return df_train_quality, df_test_cls, df_val_cls







            # 0) Split data---- Good but not perfect in classification ---- (1)
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
            candidate_ks = [10, 20, 40, 50, 60, 70]
            #candidate_ks = [10, 20, 40, 50 ]
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

                    print(f"\n=== Nayef : Classification_report on Full FINAL TRAIN SET (n={n_quality}) ===")
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
