## from pyspark.sql.functions import col
import colorama
import math
import numpy as np
import numpy as np
# import cudf
# import cupy as cp
import numpy as np
import pandas as pd
import time
import torch
import warnings
# import sparknlp
# from pyspark.ml import Pipeline
# from pyspark.ml import Pipeline
# from pyspark.ml import Pipeline
# from pyspark.ml import Pipeline
# from pyspark.ml import Pipeline
# from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.clustering import LDA
from pyspark.ml.clustering import LDA
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer
# from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer
from pyspark.ml.feature import PCA as SparkPCA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.functions import vector_to_array
from pyspark.ml.functions import vector_to_array
# from pyspark.sql.types import VectorUDT
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame
# from transformers import pipeline
from pyspark.sql import DataFrame, functions as F
from pyspark.sql import functions as F
from pyspark.sql import functions as F
from pyspark.sql.functions import array_max, col, lit
from pyspark.sql.functions import col
from pyspark.sql.functions import col, array_max
from pyspark.sql.functions import col, array_max, coalesce
from pyspark.sql.functions import col, count
# from sparknlp.annotator import Tokenizer, Normalizer, LemmatizerModel, StopWordsCleaner
# from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, BertEmbeddings
# from sparknlp.annotator import Tokenizer, SentimentDLModel
# from sparknlp.annotator import Tokenizer, ViveknSentimentApproach
# from sparknlp.annotator import Tokenizer, WordEmbeddingsModel, SentenceEmbeddings, SentimentDLModel
# from sparknlp.annotator import Tokenizer, WordEmbeddingsModel, SentenceEmbeddings, SentimentDLModel
# from sparknlp.annotator import UniversalSentenceEncoder, SentimentDLModel
# from sparknlp.base import DocumentAssembler, Tokenizer
# from sparknlp.annotator import WordEmbeddingsModel, SentenceEmbeddings, SentimentDLModel
# from sparknlp.annotator import WordEmbeddingsModel, SentimentDLModel
# from sparknlp.base import DocumentAssembler
# from sparknlp.base import DocumentAssembler
# from sparknlp.base import DocumentAssembler, Finisher
# from sparknlp.base import DocumentAssembler, Finisher
# from sparknlp.base import LightPipeline
from pyspark.sql.functions import col, when, trim, lower, isnan, lit
from pyspark.sql.functions import expr, when, size, lit
from pyspark.sql.functions import lit, col, coalesce, udf, when, size
from pyspark.sql.functions import round, col
from pyspark.sql.functions import split, length, udf, col, size, array
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import year, month, dayofmonth, hour, minute, second
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.types import DoubleType
# from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.sql.types import StringType
from pyspark.sql.types import (StructType, StructField, StringType)
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA as sklearnPCA
from transformers import pipeline

warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class FeaturesExtractor:

    @staticmethod
    def features_extracting_configuring_tuning(features_extracting_obj, doc_topic_df_path, sentiment_df_path,
                                               Dataset_name, pre_final_global_features_pkl_path, df_features, spark):
        """
        Main pipeline for feature extraction and configuration

        Parameters:
        -----------
        features_extracting_obj : FeaturesExtractor instance
        doc_topic_df_path : str, path to save document-topic dataframe
        sentiment_df_path : str, path to save sentiment analysis results
        Dataset_name : str, name of the dataset being processed
        pre_final_global_features_pkl_path : str, path for intermediate feature storage
        df_features : DataFrame, input data containing log messages

        Returns:
        --------
        tuple: (number_of_components, best_topic_number)
        """

        # ==================== DATA VALIDATION AND PREPARATION ====================
        print("[INFO] DataFrame schema and sample:")
        df_features.printSchema()
        count_row_notnull = df_features.filter(col("processed_EventTemplate").isNotNull()).count()
        count_row_null = df_features.filter(col("processed_EventTemplate").isNull()).count()
        print(f"✅ Rows with processed_EventTemplate notnull: {count_row_notnull}")
        print(f"✅ Rows with processed_EventTemplate null: {count_row_null}")
        #        exit()

        df_features = df_features.filter(col("processed_EventTemplate").isNotNull())
        df_features_count = df_features.count()

        # ✅ Sort by Node_block_id and Timestamp
        df_features = df_features.orderBy(["Node_block_id", "Timestamp"])

        # ✅ Ensure processed_EventTemplate is string type
        df_features = df_features.withColumn("processed_EventTemplate", F.col("processed_EventTemplate").cast("string"))

        print("[INFO] DataFrame after preparation:")
        df_features.printSchema()
        # ==================== SENTIMENT FEATURE EXTRACTION ====================
        print("[INFO] Starting sentiment analysis feature extraction...")
        df_features_with_sentiment = features_extracting_obj.start_sentiment_extracting(Dataset_name, df_features,
                                                                                        sentiment_df_path, spark)
        df_features_with_sentiment.printSchema()
        #exit()
        # ==================== TOPIC MODELING FEATURE EXTRACTION ===============
        print("[INFO] Starting topic modeling feature extraction...")
        best_topic_number, df_features_with_sentiment_topic = features_extracting_obj.start_topics_extracting(
            Dataset_name, df_features_with_sentiment, sentiment_df_path, doc_topic_df_path, spark)
        df_features_with_sentiment_topic.printSchema()
        #exit()
        # ==================== ADDITIONAL FEATURE EXTRACTION ===================
        print("[INFO] Extracting additional features (temporal, statistical, entropy)...")
        best_topic_number = 15    # 15 SP_150MB
        df_feature_full_dataset = features_extracting_obj.features_extracted_different_features(Dataset_name,
                                                                                                best_topic_number,
                                                                                                df_features_with_sentiment_topic,
                                                                                                doc_topic_df_path,
                                                                                                pre_final_global_features_pkl_path,
                                                                                                spark)
        df_feature_full_dataset.printSchema()
        #exit()
        # ==================== SEMANTIC FEATURE EXTRACTION =====================
        print("[INFO] Starting semantic feature extraction using BERT embeddings...")
        df_feature_full_dataset = df_features

        number_component, final_all_features_df = features_extracting_obj.start_semantic_extraction(Dataset_name, spark,
                                                                                                    df_feature_full_dataset,
                                                                                                    text_col="processed_EventTemplate",
                                                                                                    device="cpu",
                                                                                                    batch_size=32,
                                                                                                    target_variance=0.95,
                                                                                                    max_pca_k=50)
        exit()
        final_all_features_df.printSchema()
        return number_component, best_topic_number, final_all_features_df

    def start_sentiment_extracting(self, Dataset_name, df_features, sentiment_df_path, spark):
        print(f"Spark version: {spark.version}")
        print(f"Spark version: {spark.version}")

        # 1️⃣ Extract unique templates
        sdf_unique = df_features.select("processed_EventTemplate").distinct()

        # 2️⃣ Define schema for mapInPandas
        schema = StructType([StructField("processed_EventTemplate", StringType(), True),
                             StructField("sentiment_label", StringType(), True), ])

        # 3️⃣ Batch inference function
        def analyze_batch(iterator):
            sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english",
                                          device=-1,  # CPU
                                          )
            for pdf in iterator:
                texts = pdf["processed_EventTemplate"].tolist()
                results = sentiment_pipeline(texts, batch_size=16)
                pdf["sentiment_label"] = [r["label"] for r in results]
                yield pdf

            # 4️⃣ Run distributed inference

        sdf_sentiment = sdf_unique.mapInPandas(analyze_batch, schema=schema)
        sdf_sentiment.printSchema()
        df_features.printSchema()

        # 5️⃣ Join sentiment back to original Spark DataFrame
        df_features_with_sentiment = df_features.join(sdf_sentiment, on="processed_EventTemplate", how="left")
        # 6️⃣ Optional: Verify
        num_rows = df_features_with_sentiment.count()
        count_notnull = df_features_with_sentiment.filter(
            df_features_with_sentiment["sentiment_label"].isNotNull()).count()
        print(f"Total rows: {num_rows}, Rows with sentiment: {count_notnull}")
        df_features_with_sentiment.printSchema()
        #        exit()
        output_path = Dataset_name + "_sentiment_df.parquet"
        df_features_with_sentiment.write.mode("overwrite").parquet(output_path)
        print(f"✅ sentiment  saved successfully to {output_path}")
        # exit()
        return df_features_with_sentiment

    def spark_get_best_topic_number(self, spark, df_text, col="processed_EventTemplate"):
        """
        Spark-native LDA topic selection.
        """

        topic_candidates = [2, 3, 5, 7, 9, 13, 15]
        metrics = {}

        # 1) Tokenize text
        tokenizer = Tokenizer(inputCol=col, outputCol="words")
        df_words = tokenizer.transform(df_text)

        # 2) CountVectorizer
        cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize=50000, minDF=2)
        cv_model = cv.fit(df_words)
        df_cv = cv_model.transform(df_words)

        # 3) IDF
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf_model = idf.fit(df_cv)
        df_features = idf_model.transform(df_cv)

        # Evaluate different topic numbers
        for k in topic_candidates:
            lda = LDA(k=k, maxIter=50, featuresCol="features")
            model = lda.fit(df_features)

            ll = model.logLikelihood(df_features)
            perp = model.logPerplexity(df_features)

            metrics[k] = {"logLikelihood": ll, "perplexity": perp}
            print(f"[INFO] k={k}, logLikelihood={ll}, perplexity={perp}")

        # Choose best topic number: highest log-likelihood
        best_k = max(metrics, key=lambda x: metrics[x]["logLikelihood"])
        print(f"[RESULT] Best k = {best_k}")

        return best_k, cv_model, idf_model

    def start_topics_extracting(self, Dataset_name, df_features_with_sentiment, sentiment_df_path, doc_topic_df_path,
                                spark):

        output_path = Dataset_name + "_sentiment_df.parquet"
        # ✅ Load from Parquet
        sentiment_df = spark.read.parquet(output_path)
        sentiment_df.printSchema()
        #        exit()

        print("✅ Loaded DataFrame schema:")

        df_features = sentiment_df  # df_features_with_sentiment

        df_features = df_features.withColumn("processed_EventTemplate", F.col("processed_EventTemplate").cast("string"))
        print("[STATUS] Starting Spark-native topic modeling...")
        # ---------------------------------------------------------
        # 0) Extract distinct texts for efficiency
        # ---------------------------------------------------------
        df_unique = df_features.select("processed_EventTemplate", "Node_block_id").distinct()

        # 1) Tune topic number using Spark-native function
        best_k, cv_model, idf_model = self.spark_get_best_topic_number(spark, df_unique, col="processed_EventTemplate")

        # 2) Tokenize
        tokenizer = Tokenizer(inputCol="processed_EventTemplate", outputCol="words")
        df_words = tokenizer.transform(df_unique)

        # 3) Build features
        df_cv = cv_model.transform(df_words)
        df_features_vec = idf_model.transform(df_cv)

        # 4) Train final LDA model
        lda = LDA(k=best_k, maxIter=100, featuresCol="features")
        lda_model = lda.fit(df_features_vec)

        # 5) Extract topic distributions
        df_topics = lda_model.transform(df_features_vec)

        # Convert vector to array
        df_topics = df_topics.withColumn("topicDistribution_array", vector_to_array(F.col("topicDistribution")))

        # Compute Dominant_Topic = max probability in topic distribution
        df_topics = df_topics.withColumn("Dominant_Topic", array_max(F.col("topicDistribution_array")))

        # Expand into T_0 ... T_k
        for i in range(best_k):
            df_topics = df_topics.withColumn(f"T_{i}", F.col("topicDistribution_array")[i])

        topic_cols = [f"T_{i}" for i in range(best_k)]
        df_topics_selected = df_topics.select("processed_EventTemplate", "Node_block_id",
                                              #          "topicDistribution_array",
                                              "Dominant_Topic", *topic_cols)

        # ---------------------------------------------------------
        # 6) LEFT JOIN back to original dataframe with sentiment
        # ---------------------------------------------------------
        df_features_with_sentiment_topic = (
            df_features.join(df_topics_selected, on=["processed_EventTemplate", "Node_block_id"], how="left"))
        df_features_with_sentiment_topic = df_features_with_sentiment_topic.fillna({"Dominant_Topic": 0})
        print("[SUCCESS] Final dataframe with Sentiment + Topics created!")
        num_rows = df_features_with_sentiment_topic.count()
        count_notnull = df_features_with_sentiment_topic.filter(
            df_features_with_sentiment_topic["Dominant_Topic"].isNotNull()).count()
        print(f"Total rows: {num_rows}, Rows with Dominant_Topic : {count_notnull}")
        output_path = Dataset_name + "_Topic_sentiment_df.parquet"
        df_features_with_sentiment_topic.write.mode("overwrite").parquet(output_path)
        print(f"✅ Topic_sentiment_df.parquet  saved successfully to {output_path}")

        df_features_with_sentiment_topic.printSchema()
        #        exit()

        return best_k, df_features_with_sentiment_topic

    def features_extracted_different_features(self, Dataset_name, best_topic_number, mapped_df_topics_sentiment,
                                              doc_topic_df_path, pre_final_global_features_pkl_path, spark):
        """
        Extract additional features including temporal, statistical, and entropy features

        Parameters:
        -----------
        best_topic_number : int, number of topics for feature naming
        doc_topic_df_path : str, path to topic modeling results
        pre_final_global_features_pkl_path : str, path to save enhanced features
        """

        output_path = Dataset_name + "_Topic_sentiment_df.parquet"
        # ✅ Load from Parquet
        df_features_with_sentiment_topic = spark.read.parquet(output_path)

        df_feature_full_dataset_all = df_features_with_sentiment_topic  # mapped_df_topics_sentiment  # pd.read_pickle(doc_topic_df_path)
        df_feature_full_dataset_all = df_feature_full_dataset_all.withColumn("processed_EventTemplate",
                                                                             col("processed_EventTemplate").cast(
                                                                                 "string"))
        df_feature_full_dataset = df_feature_full_dataset_all.drop("Date", "Time")
        df_feature_full_dataset = df_feature_full_dataset.withColumn("Timestamp", to_timestamp("Timestamp"))

        temporal_components = {'year': year('Timestamp'), 'month': month('Timestamp'), 'day': dayofmonth('Timestamp'),
                               'hour': hour('Timestamp'), 'minute': minute('Timestamp'), 'second': second('Timestamp')}

        for name, func in temporal_components.items():
            df_feature_full_dataset = df_feature_full_dataset.withColumn(name, func)

        print('[STATUS] Calculating text statistical features...')

        df_feature_full_dataset = df_feature_full_dataset.withColumn("num_words",
                                                                     size(split(col("processed_EventTemplate"), " ")))

        df_feature_full_dataset = df_feature_full_dataset.withColumn("Character_Count",
                                                                     length(col("processed_EventTemplate")))
        df_feature_full_dataset.printSchema()

        # ==================== TOPIC ENTROPY CALCULATION ======================
        # 1️⃣ Topic columns
        topic_cols = [f"T_{i}" for i in range(best_topic_number)]
        print(f"[INFO] Topic columns: {topic_cols}")

        def calc_entropy(topic_distribution):
            """
            Calculate entropy of topic distribution (PySpark-safe version)
            """
            if topic_distribution is None:
                return None

            entropy = 0.0
            for p in topic_distribution:
                if p > 0:
                    entropy -= p * math.log(p, 2)  # log2(p)

            return float(entropy)

        entropy_udf = udf(calc_entropy, DoubleType())
        # 3️⃣ Clean topic columns (handle None, NaN, string, or negatives)
        for c in topic_cols:
            df_feature_full_dataset = df_feature_full_dataset.withColumn(c, when(
                col(c).isNull() | isnan(col(c)) | (col(c).cast("double") < 0), lit(0.0)).otherwise(
                col(c).cast("double")))

        unique_topic_df = df_feature_full_dataset.select("Node_block_id", "processed_EventTemplate",
                                                         *topic_cols).distinct()

        unique_topic_df = unique_topic_df.withColumn("entropy", entropy_udf(array(*[col(c) for c in topic_cols])))

        df_feature_full_dataset.printSchema()
        df_feature_full_dataset = df_feature_full_dataset.select("processed_EventTemplate", "Node_block_id",
                                                                 "Timestamp", "Label", "Temp_label", "sentiment_label",
                                                                 "Dominant_Topic", "year", "month", "day", "hour",
                                                                 "minute", "second", "Character_Count", "num_words")
        unique_topic_df = unique_topic_df.select("processed_EventTemplate", "Node_block_id", "entropy")

        df_feature_full_dataset.printSchema()
        unique_topic_df.printSchema()

        # 7️⃣ Join entropy back to the main dataset
        df_feature_full_dataset = df_feature_full_dataset.join(unique_topic_df,
                                                               on=["processed_EventTemplate", "Node_block_id"],
                                                               how="left")

        print("[SUCCESS] Final dataframe with Entropy ")
        num_rows = df_feature_full_dataset.count()
        count_notnull = df_feature_full_dataset.filter(df_feature_full_dataset["entropy"].isNotNull()).count()
        print(f"Total rows: {num_rows}, Rows with entropy : {count_notnull}")

        output_path = Dataset_name + "_Topic_sentiment_diff_df.parquet"
        df_feature_full_dataset.write.mode("overwrite").parquet(output_path)
        print(f"✅ sentiment_topic_diff  saved successfully to {output_path}")

        df_feature_full_dataset.printSchema()
        #        exit()
        return df_feature_full_dataset

    # ---------------------------------------------------------
    # 1️⃣ Generate BERT embeddings
    # ---------------------------------------------------------
    def generate_bert_embeddings(self, sentences, device="cpu", batch_size=64):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        model.to(torch.device(device))
        embeddings = {}

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_emb = model.encode(batch, convert_to_tensor=False)
            for text, emb in zip(batch, batch_emb):
                embeddings[text] = emb

        return embeddings

    # ---------------------------------------------------------
    # 2️⃣ PySpark pandas_udf for distributed embedding generation
    # ---------------------------------------------------------
    def bert_embedding_udf(self, device="cpu", batch_size=64):
        @F.pandas_udf(ArrayType(FloatType()))
        def udf(texts: pd.Series) -> pd.Series:
            embeddings_map = self.generate_bert_embeddings(list(texts), device=device, batch_size=batch_size)
            return pd.Series([list(embeddings_map[t]) for t in texts])

        return udf

    # ---------------------------------------------------------
    # 3️⃣ Semantic feature extraction with PCA fine-tuning
    # ---------------------------------------------------------
    def start_semantic_extraction(self, Dataset_name, spark, sdf_features, text_col="processed_EventTemplate",
                                  device="cpu", batch_size=64, target_variance=0.95, max_pca_k=50):

        # -------------------------------
        # 1️⃣ Deduplicate unique templates
        # -------------------------------

        output_path = Dataset_name + "_Topic_sentiment_diff_df.parquet"

        # ✅ Load from Parquet
        df_features_with_sentiment_topic_diff = spark.read.parquet(output_path)
        sdf_features = df_features_with_sentiment_topic_diff

        sdf_unique_cases_blocks = sdf_features.select("processed_EventTemplate", "Node_block_id").distinct()
        sdf_unique = sdf_features.select("processed_EventTemplate").distinct()

        print("[INFO] Unique templates:", sdf_unique.count())
        print("[INFO] Unique blocks:", sdf_unique_cases_blocks.count())

        # -------------------------------
        # 2️⃣ Compute distributed BERT embeddings
        # -------------------------------
        udf_embedding = self.bert_embedding_udf(device=device, batch_size=batch_size)
        sdf_unique = sdf_unique.withColumn("embedding", udf_embedding(F.col(text_col)))

        # -------------------------------
        # 3️⃣ Convert embeddings to VectorUDT for PCA
        # -------------------------------
        to_vector_udf = F.udf(lambda arr: Vectors.dense(arr), VectorUDT())
        sdf_unique = sdf_unique.withColumn("vector_emb", to_vector_udf(F.col("embedding")))

        # -------------------------------
        # 4️⃣ Sample subset for fast PCA tuning
        # -------------------------------
        sample_fraction = 0.8  # 10% of unique templates
        seed = 42
        sdf_sample = sdf_unique.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
        print("[INFO] Sample size for PCA tuning:", sdf_sample.count())
        # exit()
        # -------------------------------
        # 5️⃣ Fine-tune PCA: test only specific values
        # -------------------------------
        candidate_ks = [10, 20, 40, 50]
        best_k = candidate_ks[-1]

        # Step 1️⃣: collect embeddings locally
        pdf = sdf_sample.select("vector_emb").toPandas()

        # Convert list → numpy array
        X = np.vstack(pdf["vector_emb"].values)

        for k in candidate_ks:
            print(f"[INFO] Testing PCA with k={k}")
            pca = sklearnPCA(n_components=k)
            model = pca.fit(X)  # use sampled subset
            explained_variance = float(sum(model.explained_variance_ratio_))

            print(f"[INFO] PCA k={k}, explained variance={explained_variance:.4f}")
            if explained_variance >= target_variance:
                best_k = k
                break

        print(f"[RESULT] Selected PCA components (best_k): {best_k}")
        #        exit()
        # -------------------------------
        # 6️⃣ Apply final PCA on all unique templates
        # -------------------------------
        pca_final = SparkPCA(k=best_k, inputCol="vector_emb", outputCol="reduced_vector")
        pca_model = pca_final.fit(sdf_unique)
        sdf_unique = pca_model.transform(sdf_unique)

        sdf_unique_cases_blocks = sdf_unique_cases_blocks.join(sdf_unique, on="processed_EventTemplate", how="left")
        print("[INFO] Unique templates:", sdf_unique.count())
        print("[INFO] Unique blocks:", sdf_unique_cases_blocks.count())
        #        exit()

        print('# Convert vector to array------')
        vector_to_array_udf = F.udf(lambda v: v.toArray().tolist() if v is not None else [0.0] * best_k,
                                    ArrayType(FloatType()))
        sdf_unique_cases_blocks = sdf_unique_cases_blocks.withColumn("reduced_embedding",
                                                                     vector_to_array_udf(F.col("reduced_vector")))

        # Select only necessary columns for join
        sdf_unique_cases_blocks_final = sdf_unique_cases_blocks.select("processed_EventTemplate", "Node_block_id",
                                                                       "reduced_embedding")
        # -------------------------------
        print(' back to original dataframe------')

        sdf_features.printSchema()
        sdf_unique_cases_blocks_final.printSchema()
        # exit()

        sdf_final = sdf_features.join(sdf_unique_cases_blocks_final, on=["processed_EventTemplate", "Node_block_id"],
                                      how="left")
        sdf_final.printSchema()

        output_path = Dataset_name + "_Topic_sentiment_diff_semantic_df.parquet"
        sdf_final.write.mode("overwrite").parquet(output_path)
        print(f"✅ Topic_sentiment_diff_semanti  saved successfully to {output_path}")

        num_rows = sdf_final.count()
        count_notnull = sdf_final.filter(sdf_final["reduced_embedding"].isNotNull()).count()
        print(f"Total rows: {num_rows}, Rows with embedding :  {count_notnull}")

        return best_k, sdf_final
