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
#from kneed import KneeLocator
#from kneed import KneeLocator
#from kneed import KneeLocator
#from kneed import KneeLocator
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
from pyspark.storagelevel import StorageLevel
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
    def features_aggregation_transformation(bert_component,final_train_with_test_with_val, dataset):

        print("---- Starting feature aggregation and transformation ----")
        final_train_with_test_with_val.printSchema()

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

        return bert_component, summ_train_test_val_combine_scaled, X_sequences_df, y_sequences_df

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
    def novelty_detection_label_establishment(DATASET, sequences_df, spark):

        from pyspark.sql.functions import col
        from pyspark.sql import functions as F
        from pyspark.sql.functions import col, lit, when
        from pyspark.ml.functions import vector_to_array
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from sklearn.metrics import precision_recall_curve
        from pyspark.sql.functions import when, lower, trim, col

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

        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, lit, when, lower, trim, udf
        from pyspark.sql.types import DoubleType
        from pyspark.ml.feature import StandardScaler, PCA as SparkPCA, VectorAssembler
        from pyspark.ml.clustering import GaussianMixture
        from pyspark.ml.classification import GBTClassifier
        from sklearn.metrics import classification_report, f1_score
        import numpy as np
        from pyspark.sql.functions import col, lit, when, abs as ps_abs, hash as ps_hash, pmod
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import GBTClassifier
        from pyspark.ml.functions import vector_to_array
        from pyspark.ml.classification import FMClassifier

        import numpy as np
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

        feature_col = "features_vec_final"
        id_col = "Node_block_id"

        # ======================================
        # 3) StandardScaler : we do not need scaling becuase we scaled after aggregation
        # ======================================
        #scaler = StandardScaler(inputCol=feature_col, outputCol="features_scaled", withMean=True, withStd=True)
        #scaler_model = scaler.fit(train_normal_df)
        #train_normal_scaled = scaler_model.transform(train_normal_df)
        #train_unlabeled_scaled = scaler_model.transform(train_unlabeled_df)
        train_normal_scaled =train_normal_df
        train_unlabeled_scaled =train_unlabeled_df
        # ======================================
        # 4) PCA for dimensionality reduction
        # ======================================
        candidate_ks = [10, 20, 40, 50, 60]
        target_variance = 0.999
        best_k = None

        for k in candidate_ks:
            #pca_tmp = SparkPCA(k=k, inputCol="features_scaled", outputCol=f"pca_features_k{k}")
            pca_tmp = SparkPCA(k=k, inputCol="features_vec_final", outputCol=f"pca_features_k{k}")
            pca_tmp_model = pca_tmp.fit(train_normal_scaled)
            explained_variance = float(sum(pca_tmp_model.explainedVariance))
            if explained_variance >= target_variance:
                best_k = k
                break
        if best_k is None:
            best_k = candidate_ks[-1]

        #pca = SparkPCA(k=best_k, inputCol="features_scaled", outputCol="pca_features")
        pca = SparkPCA(k=best_k, inputCol="features_vec_final", outputCol="pca_features")
        pca_model = pca.fit(train_normal_scaled)

        train_normal_pca = pca_model.transform(train_normal_scaled)
        train_unlabeled_pca = pca_model.transform(train_unlabeled_scaled)

        # ======================================
        # 5) PCA reconstruction error (optional hybrid score)
        # ======================================
        pc = pca_model.pc.toArray()
        #d = len(train_normal_scaled.select("features_scaled").head()[0])
        d = len(train_normal_scaled.select("features_vec_final").head()[0])

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
                                                      # reconstruction_error(col("features_scaled"),
                                                       reconstruction_error(col("features_vec_final"),
                                                                                col("pca_features")))

        train_unlabeled_pca = train_unlabeled_pca.withColumn("anomaly_score_pca",
                                                            # reconstruction_error(col("features_scaled"),
                                                            reconstruction_error(col("features_vec_final"),
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
        print("\n===Novelty Detection  Classification_report on FULL TRAIN ===")
        print(classification_report(pdf_full["true_label"], pdf_full["Final_Label"], digits=3))


        return df_full_train_labeled_features , sequences_df




















