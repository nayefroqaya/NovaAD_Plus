import warnings
import time
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
    def novelty_detection_label_establishment(DATASET, sequences_df: DataFrame, spark: SparkSession, method: str = "gmm"
                                              ):

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

            import time
            import numpy as np
            from pyspark.sql.functions import col, lit, when, pmod
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.classification import GBTClassifier
            from pyspark.ml.functions import vector_to_array
            from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

            from pyspark.sql.functions import hash as ps_hash, abs as ps_abs



            import numpy as np
            from pyspark.sql import functions as F
            from pyspark.sql.functions import col, lit, when, lower, trim, udf
            from pyspark.sql.types import DoubleType
            from pyspark.ml.feature import PCA as SparkPCA
            from pyspark.ml.clustering import GaussianMixture
            from pyspark.ml.functions import vector_to_array
            from sklearn.metrics import classification_report

            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F
            from pyspark.sql.functions import col, when, lit, lower, trim, udf
            from pyspark.sql.types import DoubleType
            from pyspark.ml.feature import PCA as SparkPCA, StandardScaler
            from pyspark.ml.clustering import GaussianMixture
            from pyspark.ml.stat import Summarizer
            import numpy as np
            from sklearn.metrics import classification_report
            from pyspark.sql import functions as F
            from pyspark.ml.classification import LogisticRegression
            from pyspark.ml.evaluation import BinaryClassificationEvaluator
            from pyspark.ml.functions import vector_to_array
            from sklearn.metrics import classification_report, f1_score
            import numpy as np

            from pyspark.sql.functions import col, when, lit
            from pyspark.ml.classification import GBTClassifier
            from pyspark.ml.evaluation import BinaryClassificationEvaluator
            from sklearn.metrics import classification_report
            import pandas as pd

            from pyspark.sql.functions import col, when, lit
            from pyspark.ml.classification import GBTClassifier
            from pyspark.ml.evaluation import BinaryClassificationEvaluator
            from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
            from pyspark.sql.functions import when

            from pyspark.sql.functions import col, lit, explode, array_repeat
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.classification import GBTClassifier
            from pyspark.ml.functions import vector_to_array
            from sklearn.metrics import f1_score, classification_report
            import numpy as np
            import time

            from pyspark.sql import SparkSession
            from pyspark.sql.functions import col, lit, when, lower, trim, udf
            from pyspark.sql.types import DoubleType
            from pyspark.ml.feature import StandardScaler, PCA as SparkPCA, VectorAssembler
            from pyspark.ml.clustering import GaussianMixture
            from pyspark.ml.classification import GBTClassifier
            from sklearn.metrics import classification_report, f1_score
            import numpy as np
            import time
            from pyspark.sql.functions import col, lit, when, abs as ps_abs, hash as ps_hash, pmod
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.classification import GBTClassifier
            from pyspark.ml.functions import vector_to_array

            import numpy as np
            import time
            from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

            # ======================================Good ND and classification AD
            # 1) Prepare y_true and Label
            # ======================================
            GT_COL = "Label"
            sequences_df = sequences_df.withColumn("y_true",
                when(lower(trim(col(GT_COL))).isin("anomaly", "1", "true", "yes"), lit(1)).when(
                    lower(trim(col(GT_COL))).isin("normal", "0", "false", "no"), lit(0)).otherwise(
                    col(GT_COL).cast("int")))

            sequences_df = sequences_df.withColumn("Label",
                when(col("Label") == "normal", 0).when(col("Label") == "anomaly", 1).otherwise(None))

            # ======================================
            # 2) Split TRAIN (Temp_label 0 or 999)
            # ======================================
            train_seq_df = sequences_df.filter(col("Temp_label").isin([0, 999])).cache()
            train_normal_df = train_seq_df.filter(col("Label") == 0)
            train_unlabeled_df = train_seq_df.filter(col("Temp_label") == 999)
            test_df = train_seq_df.filter(col("Temp_label") == 888)

            feature_col = "features_vec_final"
            id_col = "Node_block_id"

            # ======================================
            # 3) StandardScaler
            # ======================================
            scaler = StandardScaler(inputCol=feature_col, outputCol="features_scaled", withMean=True, withStd=True)
            scaler_model = scaler.fit(train_normal_df)
            train_normal_scaled = scaler_model.transform(train_normal_df)
            train_unlabeled_scaled = scaler_model.transform(train_unlabeled_df)

            # ======================================
            # 4) PCA for dimensionality reduction
            # ======================================
            candidate_ks = [10, 20, 40, 50, 60]
            target_variance = 0.999
            best_k = None

            for k in candidate_ks:
                pca_tmp = SparkPCA(k=k, inputCol="features_scaled", outputCol=f"pca_features_k{k}")
                pca_tmp_model = pca_tmp.fit(train_normal_scaled)
                explained_variance = float(sum(pca_tmp_model.explainedVariance))
                if explained_variance >= target_variance:
                    best_k = k
                    break
            if best_k is None:
                best_k = candidate_ks[-1]

            pca = SparkPCA(k=best_k, inputCol="features_scaled", outputCol="pca_features")
            pca_model = pca.fit(train_normal_scaled)

            train_normal_pca = pca_model.transform(train_normal_scaled)
            train_unlabeled_pca = pca_model.transform(train_unlabeled_scaled)

            # ======================================
            # 5) PCA reconstruction error (optional hybrid score)
            # ======================================
            pc = pca_model.pc.toArray()
            d = len(train_normal_scaled.select("features_scaled").head()[0])
            pc_b = spark.sparkContext.broadcast(pc)
            use_pc_dk_b = spark.sparkContext.broadcast(pc.shape[0] == d)

            @udf(DoubleType())
            def reconstruction_error(orig_vec, pca_vec):
                x = np.array(orig_vec.toArray(), dtype=float)
                z = np.array(pca_vec.toArray(), dtype=float)
                pc_local = pc_b.value
                x_hat = (pc_local @ z) if use_pc_dk_b.value else (z @ pc_local)
                return float(np.linalg.norm(x - x_hat))

            train_normal_pca = train_normal_pca.withColumn("anomaly_score_pca",
                                                           reconstruction_error(col("features_scaled"),
                                                                                col("pca_features")))
            train_unlabeled_pca = train_unlabeled_pca.withColumn("anomaly_score_pca",
                                                                 reconstruction_error(col("features_scaled"),
                                                                                      col("pca_features")))

            # ======================================
            # 6) Fit GMM on normal PCA space
            # ======================================
            gmm_candidates = [2, 4, 6, 8, 10, 12]
            best_gmm_model = None
            best_bic = float("inf")
            best_k_gmm = None

            def bic(model, df, k, d):
                try:
                    ll = float(model.summary.logLikelihood)
                except:
                    return float("inf")
                n = df.count()
                p = (k - 1) + k * d + k * (d * (d + 1) // 2)
                return -2.0 * ll + p * np.log(max(n, 1))

            d_pca = best_k
            feat_df = train_normal_pca.select("pca_features")

            for k in gmm_candidates:
                gmm = GaussianMixture(k=k, seed=123, featuresCol="pca_features", predictionCol="cluster")
                model = gmm.fit(feat_df)
                score = bic(model, feat_df, k, d_pca)
                if score < best_bic:
                    best_bic = score
                    best_gmm_model = model
                    best_k_gmm = k

            print(f"[GMM] Selected k={best_k_gmm} by BIC")

            # ======================================
            # 7) Compute NLL
            # ======================================
            weights = np.array(best_gmm_model.weights)
            means = np.stack([np.array(g.mean.toArray()) for g in best_gmm_model.gaussians], axis=0)
            covs = np.stack([np.array(g.cov.toArray()) for g in best_gmm_model.gaussians], axis=0)
            JITTER = 1e-6
            covs += np.eye(d_pca)[None, :, :] * JITTER
            inv_covs = np.linalg.inv(covs)
            sign, logdets = np.linalg.slogdet(covs)
            const = d_pca * np.log(2 * np.pi)

            bc_params = spark.sparkContext.broadcast(
                {"weights": weights, "means": means, "inv_covs": inv_covs, "logdets": logdets, "const": const})

            @udf(DoubleType())
            def gmm_nll(pca_vec):
                z = np.array(pca_vec.toArray(), dtype=float)
                P = bc_params.value
                w, m, ic, ld, cst = P["weights"], P["means"], P["inv_covs"], P["logdets"], P["const"]
                logps = []
                for i in range(len(w)):
                    diff = z - m[i]
                    quad = float(diff.T @ ic[i] @ diff)
                    logN = -0.5 * (quad + float(ld[i]) + cst)
                    logps.append(np.log(max(float(w[i]), 1e-300)) + logN)
                a = float(np.max(logps))
                logp = a + float(np.log(np.sum(np.exp(np.array(logps) - a))))
                return float(-logp)

            train_normal_pca = train_normal_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))
            train_unlabeled_pca = train_unlabeled_pca.withColumn("anomaly_score_gmm", gmm_nll(col("pca_features")))

            # ======================================
            # 8) Thresholds for hybrid labeling
            # ======================================
            TARGET_FPR = 0.05  # higher FPR to improve anomaly recall
            thr_gmm = train_normal_pca.approxQuantile("anomaly_score_gmm", [1 - TARGET_FPR], 1e-6)[0]
            thr_pca = train_normal_pca.approxQuantile("anomaly_score_pca", [1 - TARGET_FPR], 1e-6)[0]

            train_unlabeled_pca = train_unlabeled_pca.withColumn("Final_Label",
                when((col("anomaly_score_gmm") > lit(thr_gmm)) | (col("anomaly_score_pca") > lit(thr_pca)),
                     1).otherwise(0))

            # ======================================
            # 9) Build full labeled train dataset
            # ======================================
            df_normal_labeled_features = train_normal_df.select(col(id_col), col(feature_col).alias(
                "features_vec_final")).withColumn("Final_Label", lit(0).cast("int"))
            df_unlabeled_labeled_features = train_unlabeled_pca.select(col(id_col),
                                                                       col(feature_col).alias("features_vec_final"),
                                                                       col("Final_Label"))
            df_full_train_labeled_features = df_normal_labeled_features.unionByName(df_unlabeled_labeled_features)

            # ======================================
            # 10) Classification report on full training
            # ======================================
            df_full_eval = df_full_train_labeled_features.join(
                sequences_df.select(col(id_col), col("y_true").alias("true_label")), on=id_col, how="inner").select(
                "true_label", "Final_Label").dropna()

            pdf_full = df_full_eval.toPandas()
            pdf_full["true_label"] = pdf_full["true_label"].astype(int)
            pdf_full["Final_Label"] = pdf_full["Final_Label"].astype(int)
            print("\n=== Classification_report on FULL TRAIN ===")
            print(classification_report(pdf_full["true_label"], pdf_full["Final_Label"], digits=3))



            # if DATASET=='BGL' or DATASET=='TH_1G' :
            SEED = 42

            # =========================================================
            # CASE 1
            # real normal + all pseudo
            # stronger logic + faster classifier (RandomForest)
            # =========================================================

            # --------------------------
            # 0) Train data (real normal + ALL pseudo)
            # --------------------------
            df_real_normal = sequences_df.filter(col("Temp_label") == 0).select(col("Node_block_id"),
                col("features_vec_final"), lit(0).alias("Final_Label"), lit("real").alias("src"),
                col("anomaly_score_pca") if "anomaly_score_pca" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_pca"),
                col("anomaly_score_gmm") if "anomaly_score_gmm" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_gmm"),
                col("pca_flag") if "pca_flag" in sequences_df.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in sequences_df.columns else lit(0).alias("gmm_flag"))

            df_pseudo_all = df_full_train_labeled_features.select(col("Node_block_id"), col("features_vec_final"),
                col("Final_Label").cast("int").alias("Final_Label"), lit("pseudo").alias("src"),
                col("anomaly_score_pca") if "anomaly_score_pca" in df_full_train_labeled_features.columns else lit(
                    0.0).alias("anomaly_score_pca"),
                col("anomaly_score_gmm") if "anomaly_score_gmm" in df_full_train_labeled_features.columns else lit(
                    0.0).alias("anomaly_score_gmm"),
                col("pca_flag") if "pca_flag" in df_full_train_labeled_features.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in df_full_train_labeled_features.columns else lit(0).alias("gmm_flag"))

            train_df = df_real_normal.unionByName(df_pseudo_all)

            # --------------------------
            # 0.1) Deterministic split
            # --------------------------
            train_df = train_df.withColumn("split_key", pmod(ps_abs(ps_hash(col("Node_block_id"))), lit(100)))

            train_base_df = train_df.filter(col("split_key") < 90).drop("split_key")
            val_df = train_df.filter(col("split_key") >= 90).drop("split_key")

            # --------------------------
            # 0.2) Weighting
            # --------------------------
            PSEUDO_TRUST = 0.6
            WEIGHT_CAP = 8.0

            n0 = train_df.filter(col("Final_Label") == 0).count()
            n1 = train_df.filter(col("Final_Label") == 1).count()

            raw_w1 = float(n0 / max(n1, 1)) * 0.7
            w1 = float(min(raw_w1, WEIGHT_CAP))
            w0 = 1.0

            print(f"[INFO] Case1 train counts n0={n0}, n1={n1}, raw_w1={raw_w1:.4f}, "
                  f"capped_w1={w1:.4f}, PSEUDO_TRUST={PSEUDO_TRUST}")

            train_base_df = (train_base_df.withColumn("baseClassWeight",
                                                      when(col("Final_Label") == 1, lit(w1)).otherwise(
                                                          lit(w0))).withColumn("srcWeight", when(col("src") == "pseudo",
                                                                                                 lit(PSEUDO_TRUST)).otherwise(
                lit(1.0))).withColumn("classWeight", col("baseClassWeight") * col("srcWeight")))

            val_df = (val_df.withColumn("baseClassWeight",
                                        when(col("Final_Label") == 1, lit(w1)).otherwise(lit(w0))).withColumn(
                "srcWeight", when(col("src") == "pseudo", lit(PSEUDO_TRUST)).otherwise(lit(1.0))).withColumn(
                "classWeight", col("baseClassWeight") * col("srcWeight")))

            # --------------------------
            # 0.3) Test set
            # --------------------------
            test_df = sequences_df.filter(col("Temp_label") == 888).select(col("Node_block_id"),
                col("features_vec_final"), col("y_true").alias("Final_Label"),
                col("anomaly_score_pca") if "anomaly_score_pca" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_pca"),
                col("anomaly_score_gmm") if "anomaly_score_gmm" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_gmm"),
                col("pca_flag") if "pca_flag" in sequences_df.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in sequences_df.columns else lit(0).alias("gmm_flag")).withColumn(
                "Final_Label", col("Final_Label").cast("int"))

            # --------------------------
            # 1) Assemble features
            # use only columns that exist in train/val/test
            # --------------------------
            feature_cols = ["features_vec_final"]
            common_cols = set(train_base_df.columns) & set(val_df.columns) & set(test_df.columns)

            if "anomaly_score_pca" in common_cols:
                feature_cols.append("anomaly_score_pca")
            if "anomaly_score_gmm" in common_cols:
                feature_cols.append("anomaly_score_gmm")

            print(f"[INFO] Case1 feature_cols used by assembler: {feature_cols}")

            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_augmented")

            train_base_df = assembler.transform(train_base_df)
            val_df = assembler.transform(val_df)
            test_df = assembler.transform(test_df)

            features_col = "features_augmented"

            # --------------------------
            # 1.1) CACHE + MATERIALIZE
            # --------------------------
            train_base_df = train_base_df.cache()
            val_df = val_df.cache()
            test_df = test_df.cache()

            _ = train_base_df.count()
            _ = val_df.count()
            _ = test_df.count()

            # --------------------------
            # 2) Train FAST RandomForest
            # --------------------------
            #**rf = RandomForestClassifier(featuresCol=features_col, labelCol="Final_Label", weightCol="classWeight",
            #**    numTrees=80,  # lower to 50 for more speed
            #**    maxDepth=8,  # lower to 6 for more speed
            #**    maxBins=64, minInstancesPerNode=5, subsamplingRate=0.8, featureSubsetStrategy="sqrt", seed=SEED)
            rf = RandomForestClassifier(featuresCol=features_col, labelCol="Final_Label", weightCol="classWeight",
                numTrees=150, maxDepth=12, maxBins=128, minInstancesPerNode=2, minInfoGain=0.0,
                minWeightFractionPerNode=0.0, subsamplingRate=0.9, featureSubsetStrategy="all", bootstrap=True,
                seed=SEED)

            t0 = time.time()
            model = rf.fit(train_base_df)
            end_t0 = time.time()
            print(f"[INFO] Case1 : RF fit time: {(end_t0 - t0) / 60:.2f} minutes")

            # --------------------------
            # 3) VAL: tune threshold by anomaly class
            # --------------------------
            val_pred = model.transform(val_df)
            val_pdf = val_pred.select(col("Final_Label").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("prob_1"),
                col("pca_flag") if "pca_flag" in val_pred.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in val_pred.columns else lit(0).alias("gmm_flag")).toPandas()

            y_val = val_pdf["y"].values.astype(int)
            p_val = val_pdf["prob_1"].values
            pca_v = val_pdf["pca_flag"].values.astype(int) if "pca_flag" in val_pdf.columns else np.zeros_like(y_val)
            gmm_v = val_pdf["gmm_flag"].values.astype(int) if "gmm_flag" in val_pdf.columns else np.zeros_like(y_val)

            TARGET_RECALL = 0.93
            best_threshold, best_prec = 0.5, -1.0

            for t in np.arange(0.01, 0.999, 0.005):
                preds = (p_val >= t).astype(int)
                r = recall_score(y_val, preds, pos_label=1)
                if r >= TARGET_RECALL:
                    p = precision_score(y_val, preds, pos_label=1, zero_division=0)
                    if p > best_prec:
                        best_prec = p
                        best_threshold = float(t)

            if best_prec < 0:
                best_threshold, best_f1 = 0.5, -1.0
                for t in np.arange(0.01, 0.999, 0.005):
                    preds = (p_val >= t).astype(int)
                    f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = float(t)
                print(f"[INFO] Case1 threshold by best class-1 F1: t={best_threshold:.3f}, F1={best_f1:.4f}")
            else:
                print(
                    f"[INFO] Case1 threshold by precision@recall >= {TARGET_RECALL}: t={best_threshold:.3f}, precision={best_prec:.4f}")

            if (p_val >= best_threshold).sum() == 0:
                best_threshold = float(np.quantile(p_val, 0.90))
                print(
                    f"[WARN] Case1 threshold produced 0 anomalies on VAL. Using quantile fallback t={best_threshold:.6f}")

            # --------------------------
            # 3.1) VAL: auto-tune gate offset
            # --------------------------
            offset_grid = np.arange(0.04, 0.21, 0.02)
            best_offset, best_f1_gate = 0.10, -1.0

            for off in offset_grid:
                gate_t = min(best_threshold + float(off), 0.999)
                gated_preds = ((p_val >= best_threshold) | (((pca_v + gmm_v) >= 1) & (p_val >= gate_t))).astype(int)

                f1g = f1_score(y_val, gated_preds, pos_label=1, zero_division=0)
                if f1g > best_f1_gate:
                    best_f1_gate = f1g
                    best_offset = float(off)

            print(f"[INFO] Case1 best offset on VAL: {best_offset:.3f} (VAL class-1 F1={best_f1_gate:.4f})")

            # --------------------------
            # 4) TEST: gated ensemble using tuned offset
            # --------------------------
            t1 = time.time()
            test_pred = model.transform(test_df)
            end_t1 = time.time()
            print(f"[INFO] Case1 : RF predict time: {(end_t1 - t1) / 60:.2f} minutes")

            case1_test_pdf = test_pred.select(col("Final_Label").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("prob_1"), col("pca_flag"),
                col("gmm_flag")).toPandas()

            p_test = case1_test_pdf["prob_1"].values
            case1_best_threshold = best_threshold
            case1_best_offset = best_offset
            case1_gate_t = min(case1_best_threshold + case1_best_offset, 0.999)

            case1_test_pdf["final_pred"] = ((p_test >= case1_best_threshold) | (
                        ((case1_test_pdf["pca_flag"].values + case1_test_pdf["gmm_flag"].values) >= 1) & (
                            p_test >= case1_gate_t))).astype(int)

            print("\n================Case1: TEST CLASSIFICATION REPORT (FAST RF + GATED ENSEMBLE) ================")
            print(classification_report(case1_test_pdf["y"].astype(int), case1_test_pdf["final_pred"], digits=4))
            print(
                f"[INFO] Case1 best_threshold={case1_best_threshold:.4f}, best_offset={case1_best_offset:.3f}, gate_t={case1_gate_t:.4f}")

            # =========================================================
            # CASE 2
            # no oversampling
            # class weights instead
            # threshold tuned for anomaly class
            # gated ensemble instead of weak >=1 fusion
            # faster classifier (RandomForest)
            # =========================================================

            # --------------------------
            # 0) Prepare training and test sets
            # --------------------------
            df_train_normal = sequences_df.filter(col("Temp_label") == 0).select(col("Node_block_id"),
                col("features_vec_final"), lit(0).alias("Final_Label"), lit("real").alias("src"),
                col("anomaly_score_pca") if "anomaly_score_pca" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_pca"),
                col("anomaly_score_gmm") if "anomaly_score_gmm" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_gmm"),
                col("pca_flag") if "pca_flag" in sequences_df.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in sequences_df.columns else lit(0).alias("gmm_flag"))

            df_pseudo_anomalies = df_full_train_labeled_features.filter(col("Final_Label") == 1).select(
                col("Node_block_id"), col("features_vec_final"), col("Final_Label").cast("int").alias("Final_Label"),
                lit("pseudo").alias("src"),
                col("anomaly_score_pca") if "anomaly_score_pca" in df_full_train_labeled_features.columns else lit(
                    0.0).alias("anomaly_score_pca"),
                col("anomaly_score_gmm") if "anomaly_score_gmm" in df_full_train_labeled_features.columns else lit(
                    0.0).alias("anomaly_score_gmm"),
                col("pca_flag") if "pca_flag" in df_full_train_labeled_features.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in df_full_train_labeled_features.columns else lit(0).alias("gmm_flag"))

            train_df = df_train_normal.unionByName(df_pseudo_anomalies)

            # --------------------------
            # 0.1) Deterministic split
            # --------------------------
            train_df = train_df.withColumn("split_key", pmod(ps_abs(ps_hash(col("Node_block_id"))), lit(100)))

            train_base_df = train_df.filter(col("split_key") < 90).drop("split_key")
            val_df = train_df.filter(col("split_key") >= 90).drop("split_key")

            # --------------------------
            # 0.2) Weighting instead of oversampling
            # --------------------------
            PSEUDO_TRUST = 0.6
            WEIGHT_CAP = 8.0

            n0 = train_df.filter(col("Final_Label") == 0).count()
            n1 = train_df.filter(col("Final_Label") == 1).count()

            raw_w1 = float(n0 / max(n1, 1)) * 0.7
            w1 = float(min(raw_w1, WEIGHT_CAP))
            w0 = 1.0

            print(f"[INFO] Case2 train counts n0={n0}, n1={n1}, raw_w1={raw_w1:.4f}, "
                  f"capped_w1={w1:.4f}, PSEUDO_TRUST={PSEUDO_TRUST}")

            train_base_df = (train_base_df.withColumn("baseClassWeight",
                                                      when(col("Final_Label") == 1, lit(w1)).otherwise(
                                                          lit(w0))).withColumn("srcWeight", when(col("src") == "pseudo",
                                                                                                 lit(PSEUDO_TRUST)).otherwise(
                lit(1.0))).withColumn("classWeight", col("baseClassWeight") * col("srcWeight")))

            val_df = (val_df.withColumn("baseClassWeight",
                                        when(col("Final_Label") == 1, lit(w1)).otherwise(lit(w0))).withColumn(
                "srcWeight", when(col("src") == "pseudo", lit(PSEUDO_TRUST)).otherwise(lit(1.0))).withColumn(
                "classWeight", col("baseClassWeight") * col("srcWeight")))

            # --------------------------
            # 0.3) Test set
            # --------------------------
            test_df = sequences_df.filter(col("Temp_label") == 888).select(col("Node_block_id"),
                col("features_vec_final"), col("y_true").alias("Final_Label"),
                col("anomaly_score_pca") if "anomaly_score_pca" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_pca"),
                col("anomaly_score_gmm") if "anomaly_score_gmm" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_gmm"),
                col("pca_flag") if "pca_flag" in sequences_df.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in sequences_df.columns else lit(0).alias("gmm_flag")).withColumn(
                "Final_Label", col("Final_Label").cast("int"))

            # --------------------------
            # 1) Assemble features
            # use only columns that exist in train/val/test
            # --------------------------
            feature_cols = ["features_vec_final"]
            common_cols = set(train_base_df.columns) & set(val_df.columns) & set(test_df.columns)

            if "anomaly_score_pca" in common_cols:
                feature_cols.append("anomaly_score_pca")
            if "anomaly_score_gmm" in common_cols:
                feature_cols.append("anomaly_score_gmm")

            print(f"[INFO] Case2 feature_cols used by assembler: {feature_cols}")

            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_augmented")

            train_base_df = assembler.transform(train_base_df)
            val_df = assembler.transform(val_df)
            test_df = assembler.transform(test_df)

            features_col = "features_augmented"

            # --------------------------
            # 1.1) CACHE + MATERIALIZE
            # --------------------------
            train_base_df = train_base_df.cache()
            val_df = val_df.cache()
            test_df = test_df.cache()

            _ = train_base_df.count()
            _ = val_df.count()
            _ = test_df.count()

            # --------------------------
            # 2) Train FAST RandomForestClassifier
            # --------------------------
            #** rf = RandomForestClassifier(featuresCol=features_col, labelCol="Final_Label", weightCol="classWeight",
            # **    numTrees=80,  # lower to 50 for more speed
            #**    maxDepth=8,  # lower to 6 for more speed
            #**    maxBins=64, minInstancesPerNode=5, subsamplingRate=0.8, featureSubsetStrategy="sqrt", seed=123)

            rf = RandomForestClassifier(featuresCol=features_col, labelCol="Final_Label", weightCol="classWeight",
                numTrees=150, maxDepth=12, maxBins=128, minInstancesPerNode=2, minInfoGain=0.0,
                minWeightFractionPerNode=0.0, subsamplingRate=0.9, featureSubsetStrategy="all", bootstrap=True,
                seed=SEED)

            start_fit_classification = time.time()
            model = rf.fit(train_base_df)
            end_fit_classification = time.time()
            case2_Classification_time = (end_fit_classification - start_fit_classification) / 60
            print(f"Case2 : final Model classification completed in {case2_Classification_time:.2f} minutes")

            # --------------------------
            # 3) Validation predictions and threshold tuning
            # tune for anomaly class
            # --------------------------
            val_pred = model.transform(val_df)

            val_pdf = val_pred.select(col("Final_Label").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("prob_1"),
                col("pca_flag") if "pca_flag" in val_pred.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in val_pred.columns else lit(0).alias("gmm_flag")).toPandas()

            y_val = val_pdf["y"].values.astype(int)
            p_val = val_pdf["prob_1"].values
            pca_v = val_pdf["pca_flag"].values.astype(int) if "pca_flag" in val_pdf.columns else np.zeros_like(y_val)
            gmm_v = val_pdf["gmm_flag"].values.astype(int) if "gmm_flag" in val_pdf.columns else np.zeros_like(y_val)

            TARGET_RECALL = 0.93
            best_threshold, best_prec = 0.5, -1.0

            for t in np.arange(0.01, 0.999, 0.005):
                preds = (p_val >= t).astype(int)
                r = recall_score(y_val, preds, pos_label=1)
                if r >= TARGET_RECALL:
                    p = precision_score(y_val, preds, pos_label=1, zero_division=0)
                    if p > best_prec:
                        best_prec = p
                        best_threshold = float(t)

            if best_prec < 0:
                best_threshold, best_f1 = 0.5, -1.0
                for t in np.arange(0.01, 0.999, 0.005):
                    preds = (p_val >= t).astype(int)
                    f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = float(t)
                print(f"[INFO] Case2 threshold by best class-1 F1: t={best_threshold:.3f}, F1={best_f1:.4f}")
            else:
                print(
                    f"[INFO] Case2 threshold by precision@recall >= {TARGET_RECALL}: t={best_threshold:.3f}, precision={best_prec:.4f}")

            if (p_val >= best_threshold).sum() == 0:
                best_threshold = float(np.quantile(p_val, 0.90))
                print(
                    f"[WARN] Case2 threshold produced 0 anomalies on VAL. Using quantile fallback t={best_threshold:.6f}")

            # --------------------------
            # 3.1) Tune gate offset on validation
            # gated ensemble instead of weak >=1 fusion
            # --------------------------
            offset_grid = np.arange(0.04, 0.21, 0.02)
            best_offset, best_f1_gate = 0.10, -1.0

            for off in offset_grid:
                gate_t = min(best_threshold + float(off), 0.999)
                gated_preds = ((p_val >= best_threshold) | (((pca_v + gmm_v) >= 1) & (p_val >= gate_t))).astype(int)

                f1g = f1_score(y_val, gated_preds, pos_label=1, zero_division=0)
                if f1g > best_f1_gate:
                    best_f1_gate = f1g
                    best_offset = float(off)

            print(f"[INFO] Case2 best offset on VAL: {best_offset:.3f} (VAL class-1 F1={best_f1_gate:.4f})")

            # --------------------------
            # 4) Test predictions
            # --------------------------
            start_predict_classification = time.time()
            test_pred = model.transform(test_df)
            end_predict_classification = time.time()
            case2_Classification_pred_time = (end_predict_classification - start_predict_classification) / 60
            print(f"Case2 : final Model predicts completed in {case2_Classification_pred_time:.2f} minutes")

            case2_test_pdf = test_pred.select(col("Final_Label").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("prob_1"), col("pca_flag"),
                col("gmm_flag")).toPandas()

            case2_best_threshold = best_threshold
            case2_best_offset = best_offset
            case2_gate_t = min(case2_best_threshold + case2_best_offset, 0.999)

            case2_test_pdf["pred_rf"] = (case2_test_pdf["prob_1"] >= case2_best_threshold).astype(int)

            case2_test_pdf["final_pred"] = ((case2_test_pdf["prob_1"] >= case2_best_threshold) | (
                    ((case2_test_pdf["pca_flag"] + case2_test_pdf["gmm_flag"]) >= 1) & (
                        case2_test_pdf["prob_1"] >= case2_gate_t))).astype(int)

            # --------------------------
            # 5) Classification reports
            # --------------------------
            print("\n================Case1: TEST CLASSIFICATION REPORT (FAST RF + GATED ENSEMBLE) ================")
            print(classification_report(case1_test_pdf["y"].astype(int), case1_test_pdf["final_pred"], digits=4))
            print(
                f"[INFO] Case1 : best_threshold={case1_best_threshold:.4f}, best_offset={case1_best_offset:.3f}, gate_t={case1_gate_t:.4f}")
            print(f"[INFO] Case1 : RF fit time: {(end_t0 - t0) / 60:.2f} minutes")
            print(f"[INFO] Case1 : RF predict time: {(end_t1 - t1) / 60:.2f} minutes")

            print("\n================Case2: TEST CLASSIFICATION REPORT (FAST RF + STRONGER LOGIC) ================")
            print(classification_report(case2_test_pdf["y"].astype(int), case2_test_pdf["final_pred"], digits=4))
            print(
                f"[INFO] Case2 : best_threshold={case2_best_threshold:.4f}, best_offset={case2_best_offset:.3f}, gate_t={case2_gate_t:.4f}")
            print(f"Case2 : final Model classification completed in {case2_Classification_time:.2f} minutes")
            print(f"Case2 : final Model predicts completed in {case2_Classification_pred_time:.2f} minutes")

            exit()
















            # ------- classification stage. GBTClassifier-----New / Try ------------------------------------------------
            #-----------------------------------------------------------------------------------------------------------
            #-----------------------------------------------------------------------------------------------------------
            # --------------------------
            # 0) Build training data
            # --------------------------

            #if DATASET=='BGL' or DATASET=='TH_1G' :
            SEED =  42
            # --------------------------
            # 0) Train data (real normal + ALL pseudo)
            # --------------------------
            df_real_normal = sequences_df.filter(col("Temp_label") == 0).select(col("Node_block_id"),
                col("features_vec_final"), lit(0).alias("Final_Label"), lit("real").alias("src"))

            df_pseudo_all = df_full_train_labeled_features.select(col("Node_block_id"), col("features_vec_final"),
                col("Final_Label").cast("int").alias("Final_Label"), lit("pseudo").alias("src"))

            train_df = df_real_normal.unionByName(df_pseudo_all)

            # --------------------------
            # 0.1) Deterministic split (NO rand, NO randomSplit)
            # split_key in [0..99] is stable for each Node_block_id every run
            # --------------------------
            train_df = train_df.withColumn("split_key", pmod(ps_abs(ps_hash(col("Node_block_id"))), lit(100)))

            train_base_df = train_df.filter(col("split_key") < 90).drop("split_key")
            val_df = train_df.filter(col("split_key") >= 90).drop("split_key")

            # --------------------------
            # 0.2) Weighting (cap weights to reduce swings)
            # --------------------------
            PSEUDO_TRUST = 0.6
            WEIGHT_CAP = 8.0  # smaller cap = more stable, fewer crazy shifts

            n0 = train_df.filter(col("Final_Label") == 0).count()
            n1 = train_df.filter(col("Final_Label") == 1).count()

            raw_w1 = float(n0 / max(n1, 1)) * 0.7
            w1 = float(min(raw_w1, WEIGHT_CAP))
            w0 = 1.0

            print(
                f"[INFO] Train counts n0={n0}, n1={n1}, raw_w1={raw_w1:.4f}, capped_w1={w1:.4f}, PSEUDO_TRUST={PSEUDO_TRUST}")

            train_base_df = train_base_df.withColumn("baseClassWeight",
                when(col("Final_Label") == 1, lit(w1)).otherwise(lit(w0))).withColumn("srcWeight",
                when(col("src") == "pseudo", lit(PSEUDO_TRUST)).otherwise(lit(1.0))).withColumn("classWeight",
                col("baseClassWeight") * col("srcWeight"))

            val_df = val_df.withColumn("baseClassWeight",
                when(col("Final_Label") == 1, lit(w1)).otherwise(lit(w0))).withColumn("srcWeight",
                when(col("src") == "pseudo", lit(PSEUDO_TRUST)).otherwise(lit(1.0))).withColumn("classWeight",
                col("baseClassWeight") * col("srcWeight"))

            # --------------------------
            # 0.3) Test set
            # --------------------------
            test_df = sequences_df.filter(col("Temp_label") == 888).select(col("Node_block_id"),
                col("features_vec_final"), col("y_true").alias("Final_Label"),
                col("anomaly_score_pca") if "anomaly_score_pca" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_pca"),
                col("anomaly_score_gmm") if "anomaly_score_gmm" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_gmm"),
                col("pca_flag") if "pca_flag" in sequences_df.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in sequences_df.columns else lit(0).alias("gmm_flag")).withColumn(
                "Final_Label", col("Final_Label").cast("int"))

            # --------------------------
            # 1) Assemble features
            # --------------------------
            feature_cols = ["features_vec_final"]
            if "anomaly_score_pca" in train_df.columns:
                feature_cols.append("anomaly_score_pca")
            if "anomaly_score_gmm" in train_df.columns:
                feature_cols.append("anomaly_score_gmm")

            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_augmented")

            train_base_df = assembler.transform(train_base_df)
            val_df = assembler.transform(val_df)
            test_df = assembler.transform(test_df)

            features_col = "features_augmented"

            # --------------------------
            # 1.1) CACHE + MATERIALIZE (critical for stability)
            # --------------------------
            train_base_df = train_base_df.cache()
            val_df = val_df.cache()
            test_df = test_df.cache()

            _ = train_base_df.count()
            _ = val_df.count()
            _ = test_df.count()

            # --------------------------
            # 2) Train deterministic GBT (reduce internal randomness)
            # --------------------------
            #*gbt = GBTClassifier(featuresCol=features_col, labelCol="Final_Label", weightCol="classWeight",
            #*    maxIter=75, maxDepth=5, stepSize=0.1, seed=SEED, subsamplingRate=1.0, featureSubsetStrategy="all")

            gbt = RandomForestClassifier(featuresCol=features_col, labelCol="Final_Label", weightCol="classWeight",
                numTrees=75, maxDepth=5, seed=SEED, subsamplingRate=1.0, featureSubsetStrategy="all")

            t0 = time.time()
            model = gbt.fit(train_base_df)
            end_t0= time.time()
            print(f"[INFO] case1 :  GBT fit time: {(end_t0 - t0) / 60:.2f} minutes")

            # --------------------------
            # 3) VAL: tune threshold
            # --------------------------
            val_pred = model.transform(val_df)
            val_pdf = val_pred.select(col("Final_Label").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("prob_1"),
                col("pca_flag") if "pca_flag" in val_pred.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in val_pred.columns else lit(0).alias("gmm_flag")).toPandas()

            y_val = val_pdf["y"].values.astype(int)
            p_val = val_pdf["prob_1"].values
            pca_v = val_pdf["pca_flag"].values.astype(int) if "pca_flag" in val_pdf.columns else np.zeros_like(
                y_val)
            gmm_v = val_pdf["gmm_flag"].values.astype(int) if "gmm_flag" in val_pdf.columns else np.zeros_like(
                y_val)

            TARGET_RECALL =  0.93 #94 0.95
            best_threshold, best_prec = 0.9, -1.0   # 0.5

            for t in np.arange(0.01, 0.999, 0.005):
                preds = (p_val >= t).astype(int)
                r = recall_score(y_val, preds, pos_label=1)
                if r >= TARGET_RECALL:
                    p = precision_score(y_val, preds, pos_label=1, zero_division=0)
                    if p > best_prec:
                        best_prec, best_threshold = p, float(t)

            if best_prec < 0:
                best_threshold, best_f1 = 0.5, -1.0
                for t in np.arange(0.01, 0.999, 0.005):
                    preds = (p_val >= t).astype(int)
                    f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_threshold = f1, float(t)
                print(f"[INFO] Threshold by best class-1 F1: t={best_threshold:.3f}, F1={best_f1:.4f}")
            else:
                print(
                    f"[INFO] Threshold by precision@recall>= {TARGET_RECALL}: t={best_threshold:.3f}, precision={best_prec:.4f}")

            if (p_val >= best_threshold).sum() == 0:
                best_threshold = float(np.quantile(p_val, 0.90))
                print(
                    f"[WARN] Threshold produced 0 anomalies on VAL. Using quantile fallback t={best_threshold:.6f}")

            # --------------------------
            # 3.1) VAL: auto-tune gate offset (deterministic)
            # --------------------------
            offset_grid = np.arange(0.06, 0.21, 0.02)  # stable, not too wide
            best_offset, best_f1_gate = 0.12, -1.0

            for off in offset_grid:
                gate_t = min(best_threshold + float(off), 0.999)
                gated_preds = ((p_val >= best_threshold) | ((p_val >= gate_t) & ((pca_v + gmm_v) >= 1))).astype(int)
                f1g = f1_score(y_val, gated_preds, pos_label=1, zero_division=0)
                if f1g > best_f1_gate:
                    best_f1_gate, best_offset = f1g, float(off)

            print(f"[INFO] Best offset on VAL: {best_offset:.3f} (VAL class-1 F1={best_f1_gate:.4f})")

            # --------------------------
            # 4) TEST: gated ensemble using tuned offset
            # --------------------------
            t1 = time.time()
            test_pred = model.transform(test_df)
            end_t1=time.time()
            print(f"[INFO] case1 : GBT predict time: {(end_t1 - t1) / 60:.2f} minutes")

            case1_test_pdf = test_pred.select(col("Final_Label").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("prob_1"), col("pca_flag"),
                col("gmm_flag")).toPandas()

            p_test = case1_test_pdf["prob_1"].values
            gate_t = min(best_threshold + best_offset, 0.999)

            case1_test_pdf["final_pred"] = ((p_test >= best_threshold) | ((p_test >= gate_t) & (
                        (case1_test_pdf["pca_flag"].values + case1_test_pdf["gmm_flag"].values) >= 1))).astype(int)

            print("\n================Case1:  TEST CLASSIFICATION REPORT (HASH-split stable) ================")
            print(classification_report(case1_test_pdf["y"].astype(int), case1_test_pdf["final_pred"], digits=4))
            print(f"[INFO] best_threshold={best_threshold:.4f}, best_offset={best_offset:.3f}, gate_t={gate_t:.4f}")


            #else:
            # ------- classification stage. GBTClassifier-----New ------------------------------------------------------
            # ======================================
            # 0) Prepare training and test sets
            # ======================================
            df_train_normal = sequences_df.filter(col("Temp_label") == 0).select(col("Node_block_id"),
                col("features_vec_final"), lit(0).alias("Final_Label"))

            # Pseudo-labeled anomalies from novelty detection
            df_pseudo_anomalies = df_full_train_labeled_features.filter(col("Final_Label") == 1)

            # Merge normal + pseudo-labeled anomalies
            train_df = df_train_normal.unionByName(df_pseudo_anomalies)

            # Optional oversample anomalies
            n0 = train_df.filter(col("Final_Label") == 0).count()
            n1 = train_df.filter(col("Final_Label") == 1).count()
            k = int(np.ceil(n0 / max(n1, 1)))
            oversampled_anom = df_pseudo_anomalies
            for _ in range(k - 1):
                oversampled_anom = oversampled_anom.unionByName(df_pseudo_anomalies)
            train_df = train_df.unionByName(oversampled_anom)

            # Small validation split for threshold tuning (10% of train)
            train_base_df = train_df.sample(False, 0.9, seed=42)
            val_df = train_df.subtract(train_base_df)

            # Test set
            test_df = sequences_df.filter(col("Temp_label") == 888).select(col("Node_block_id"),
                col("features_vec_final"), col("y_true").alias("Final_Label"),
                col("anomaly_score_pca") if "anomaly_score_pca" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_pca"),
                col("anomaly_score_gmm") if "anomaly_score_gmm" in sequences_df.columns else lit(0.0).alias(
                    "anomaly_score_gmm"),
                col("pca_flag") if "pca_flag" in sequences_df.columns else lit(0).alias("pca_flag"),
                col("gmm_flag") if "gmm_flag" in sequences_df.columns else lit(0).alias("gmm_flag")).withColumn(
                "Final_Label", col("Final_Label").cast("int"))

            # ======================================
            # 1) Assemble features
            # ======================================
            feature_cols = ["features_vec_final"]
            if "anomaly_score_pca" in train_df.columns:
                feature_cols.append("anomaly_score_pca")
            if "anomaly_score_gmm" in train_df.columns:
                feature_cols.append("anomaly_score_gmm")

            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_augmented")
            train_base_df = assembler.transform(train_base_df)
            val_df = assembler.transform(val_df)
            test_df = assembler.transform(test_df)
            features_col = "features_augmented"

            # ======================================
            # 2) Train GBT Classifier
            # ======================================
            #*gbt = GBTClassifier(featuresCol=features_col, labelCol="Final_Label",
            #*                    maxIter=75,  # 100
            #*                    maxDepth=5, #6
            #*    stepSize=0.1, # 0.05
            #*                    seed=123)

            gbt = RandomForestClassifier(featuresCol=features_col, labelCol="Final_Label", numTrees=75, maxDepth=5,
                seed=123, subsamplingRate=1.0, featureSubsetStrategy="all")

            start_fit_classification = time.time()
            model = gbt.fit(train_base_df)
            end_fit_classification = time.time()
            case2_Classification_time = (end_fit_classification - start_fit_classification) / 60
            print(f"final Model classification  completed in {case2_Classification_time:.2f} minutes")

            # ======================================
            # 3) Validation predictions and threshold tuning
            # ======================================

            # ================================
            # Validation predictions
            # ================================
            val_pred = model.transform(val_df)

            # Convert probability vector to array
            val_pdf = val_pred.select(col("Final_Label").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("prob_1")  # class 1 probability
            ).toPandas()

            # Threshold tuning
            best_threshold = 0.5
            best_f1 = -1.0
            for t in np.arange(0.05, 0.96, 0.01):
                preds = (val_pdf["prob_1"].values >= t).astype(int)
                f1 = f1_score(val_pdf["y"].values.astype(int), preds, average="weighted")
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(t)

            print("Best threshold from VAL:", best_threshold)
            print("Best VAL weighted F1:", best_f1)

            # ================================
            # Test predictions
            # ================================

            start_predict_classification = time.time()

            test_pred = model.transform(test_df)
            end_predict_classification = time.time()
            case2_Classification_pred_time = (end_predict_classification - start_predict_classification) / 60
            print(f"final Model predicts  completed in {case2_Classification_pred_time:.2f} minutes")

            case2_test_pdf = test_pred.select(col("Final_Label").alias("y"),
                vector_to_array(col("probability")).getItem(1).alias("prob_1"),  # class 1 probability
                col("pca_flag"), col("gmm_flag")).toPandas()

            # Classifier predictions
            case2_test_pdf["pred_gbt"] = (case2_test_pdf["prob_1"] >= best_threshold).astype(int)

            # Optional ensemble: predict anomaly if >=1 signal
            case2_test_pdf["final_pred"] = ((case2_test_pdf["pred_gbt"] + case2_test_pdf["pca_flag"] + case2_test_pdf["gmm_flag"]) >= 1).astype(
                int)

            # ======================================
            # 5) Classification report
            # ======================================
            print("\n================Case1:  TEST CLASSIFICATION REPORT (HASH-split stable) ================")
            print(classification_report(case1_test_pdf["y"].astype(int), case1_test_pdf["final_pred"], digits=4))
            #print(f"[INFO] best_threshold={best_threshold:.4f}, best_offset={best_offset:.3f}, gate_t={gate_t:.4f}")


            print(f"[INFO] Case1 :  GBT fit time: {(end_t0 - t0) / 60:.2f} minutes")
            print(f"[INFO] Case1 : GBT predict time: {(end_t1 - t1) / 60:.2f} minutes")



            print("\n================Case2:  TEST CLASSIFICATION REPORT ================")
            print(classification_report(case2_test_pdf["y"].astype(int), case2_test_pdf["final_pred"], digits=4))



            print(f"Case2 : final Model classification  completed in {case2_Classification_time:.2f} minutes")
            print(f"Case2 : final Model predicts  completed in {case2_Classification_pred_time:.2f} minutes")

            exit()













