import warnings
from collections import Counter

import colorama
#import cudf
#import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import spacy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline

from datasets import Dataset as hdp

# Load English language model
nlp = spacy.load("en_core_web_lg")
warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class FeaturesExtractor:
    """
    A comprehensive feature extraction class for processing log data with multiple feature types:
    - Sentiment analysis.
    - Topic modeling (LDA) : Dominant Topic and Entropy. 
    - Semantic embeddings (SBERT).
    - Temporal features : Month, day, hour, minut, second.
    - Text statistics : Word count and characters count.
    """

    @staticmethod
    def features_extracting_configuring_tuning(features_extracting_obj,
                                               doc_topic_df_path, sentiment_df_path,
                                               Dataset_name, pre_final_global_features_pkl_path,
                                               df_features):
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
        # Data validation and preparation
        df_features.info()
        df_features.sort_values(by=['Node_block_id', 'Timestamp'], inplace=True)
        df_features = df_features.reset_index(drop=True)
        df_features['processed_EventTemplate'] = df_features['processed_EventTemplate'].astype(str)
        df_features.info()

        # ==================== SENTIMENT FEATURE EXTRACTION ====================
        print("[INFO] Starting sentiment analysis feature extraction...")
        features_extracting_obj.start_sentiment_extracting(df_features, sentiment_df_path)

        # ==================== TOPIC MODELING FEATURE EXTRACTION ===============
        print("[INFO] Starting topic modeling feature extraction...")
        best_topic_number = features_extracting_obj.start_topics_extracting(sentiment_df_path, doc_topic_df_path)

        # ==================== ADDITIONAL FEATURE EXTRACTION ===================
        print("[INFO] Extracting additional features (temporal, statistical, entropy)...")
        features_extracting_obj.features_extracted_different_features(best_topic_number,
                                                                      doc_topic_df_path,
                                                                      pre_final_global_features_pkl_path)

        # ==================== SEMANTIC FEATURE EXTRACTION =====================
        print("[INFO] Starting semantic feature extraction using BERT embeddings...")
        number_component = features_extracting_obj.start_semantic_extraction(pre_final_global_features_pkl_path)

        return number_component, best_topic_number

    @staticmethod
    def preparing_corpus_for_TM(text_features_content_list, counts=None):
        """
        Prepare text corpus for topic modeling using TF-IDF vectorization

        Parameters:
        -----------
        text_features_content_list : list of str, text documents to process
        counts : list of int, optional, frequency counts for weighted processing

        Returns:
        --------
        scipy.sparse matrix: TF-IDF transformed document-term matrix
        """
        tf_vectorizer = TfidfVectorizer(stop_words='english')
        tf_vectorizer.fit(text_features_content_list)

        import joblib
        joblib.dump(tf_vectorizer, 'tf_vectorizer.pkl')

        news_matrix = tf_vectorizer.transform(text_features_content_list)

        # Apply frequency weighting if counts are provided
        if counts is not None:
            count_matrix = sp.diags(counts)
            news_matrix = count_matrix @ news_matrix

        print(f"[DEBUG] Transformed matrix type: {type(news_matrix)}")

        return news_matrix

    def start_topics_extracting(self, sentiment_df_path, doc_topic_df_path):
        """
        Perform topic modeling using Latent Dirichlet Allocation (LDA)

        Parameters:
        -----------
        sentiment_df_path : str, path to sentiment analysis results
        doc_topic_df_path : str, path to save topic modeling results

        Returns:
        --------
        int: Optimal number of topics determined by grid search
        """
        # Load preprocessed data
        df_features = pd.read_pickle(sentiment_df_path)
        df_features.info()
        df_features['processed_EventTemplate'] = df_features['processed_EventTemplate'].astype(str)

        # Prepare text corpus
        text_feature_as_list = df_features['processed_EventTemplate'].tolist()
        print('[STATUS] Preparing corpus for topic modeling...')

        # Deduplicate log messages and count occurrences for efficiency
        log_counts = Counter(text_feature_as_list)
        unique_texts = list(log_counts.keys())
        counts = list(log_counts.values())

        # Prepare TF-IDF matrix for LDA
        cv_matrix = self.preparing_corpus_for_TM(unique_texts, counts=counts)

        # Determine optimal number of topics
        best_topic_number_val = self.get_best_topic_number(cv_matrix)
        print(f'[RESULT] Optimal number of topics: {best_topic_number_val}')

        # Perform LDA with optimal parameters
        print('[STATUS] Training LDA model with optimal parameters...')
        doc_topic_matrix = self.get_lda_topics(best_topic_number_val, cv_matrix)

        # Create topic column names
        list_topic = [f"T_{i}" for i in range(best_topic_number_val)]

        # Create topic distribution dataframe
        doc_topic_df = pd.DataFrame(doc_topic_matrix, columns=list_topic)
        doc_topic_df.info()
        print(f'[DEBUG] Document-topic matrix shape: {len(doc_topic_df)}')

        # Calculate topic distributions and dominant topics
        doc_topic_df['list_topics_distribution'] = doc_topic_df[list_topic].values.tolist()
        doc_topic_df['Dominant_Topic'] = doc_topic_df['list_topics_distribution'].apply(lambda x: max(x))

        # Map results back to original dataset
        doc_topic_dict = {msg: row for msg, row in zip(unique_texts, doc_topic_df['list_topics_distribution'])}
        df_features['list_topics_distribution'] = df_features['processed_EventTemplate'].map(doc_topic_dict)
        df_features['Dominant_Topic'] = df_features['list_topics_distribution'].apply(lambda x: max(x))

        # Expand topic distributions into individual columns
        for i, col in enumerate(list_topic):
            df_features[col] = df_features['list_topics_distribution'].apply(lambda x, idx=i: x[idx])

        # Save results
        df_features.to_pickle(doc_topic_df_path)
        df_features.info()
        print('[SUCCESS] Topic modeling completed successfully')
        print('[DEBUG] Sample topic distributions:')
        print(df_features[list_topic].head(20))

        return best_topic_number_val

    @staticmethod
    def bert_text_embedding(Dataset, sentences):
        """
        Generate BERT embeddings for text sentences

        Parameters:
        -----------
        Dataset : dataset class for creating data loader
        sentences : list of str, texts to embed

        Returns:
        --------
        dict: Mapping of sentences to their BERT embeddings
        """
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        device = torch.device('cpu')  # Force CPU usage
        print("[INFO] Using CPU for BERT encoding")
        model.to(device)

        batch_size = 64

        class SentenceDataset(Dataset):
            """Custom dataset for sentence embedding"""

            def __init__(self, sentences):
                self.sentences = sentences

            def __len__(self):
                return len(self.sentences)

            def __getitem__(self, idx):
                return self.sentences[idx]

        # Create data loader for batch processing
        dataloader = DataLoader(SentenceDataset(sentences), batch_size=batch_size, shuffle=False, num_workers=4)

        # Process sentences in batches
        embeddings = {}
        for batch in tqdm(dataloader, desc="Generating BERT embeddings"):
            batch = list(batch)
            batch_embeddings = model.encode(batch, convert_to_tensor=False)
            for text, emb in zip(batch, batch_embeddings):
                embeddings[text] = emb

        return embeddings

    def start_semantic_extraction(self, pre_final_global_features_pkl_path):
        """
        Extract semantic features using BERT embeddings and dimensionality reduction

        Parameters:
        -----------
        pre_final_global_features_pkl_path : str, path to feature dataframe

        Returns:
        --------
        int: Number of components after dimensionality reduction
        """
        print('[STATUS] Starting semantic feature extraction...')
        df_features = pd.read_pickle(pre_final_global_features_pkl_path)
        df_features.info()
        df_features['processed_EventTemplate'] = df_features['processed_EventTemplate'].astype(str)

        text_feature_as_list = df_features['processed_EventTemplate'].tolist()

        # Deduplicate to optimize embedding computation
        unique_texts = list(set(text_feature_as_list))
        print(f'[INFO] Processing {len(unique_texts)} unique sentences from {len(text_feature_as_list)} total')

        # Generate BERT embeddings
        embeddings_map = self.bert_text_embedding(Dataset, unique_texts)

        # Map embeddings back to full dataset
        embeddings_full = np.array([embeddings_map[text] for text in text_feature_as_list])

        # Apply dimensionality reduction
        print('[STATUS] Applying dimensionality reduction to embeddings...')
        reduced_embeddings, number_component = self.get_reduce_dim_on_embedding(embeddings_full)

        # Validate data integrity
        if len(df_features) != len(reduced_embeddings):
            print('[ERROR] Data length mismatch in semantic feature extraction')
            exit()

        # Store reduced embeddings
        df_features['reduced_embedding'] = list(reduced_embeddings)

        # Save results
        print('[STATUS] Saving semantic features...')
        df_features.to_pickle(pre_final_global_features_pkl_path)
        df_features.info()
        print('[SUCCESS] Semantic feature extraction completed')
        print(f'[INFO] Reduced to {number_component} components')

        return number_component

    @staticmethod
    def get_best_topic_number(cv_matrix):
        """
        Determine optimal number of topics using grid search with LDA

        Parameters:
        -----------
        cv_matrix : scipy.sparse matrix, document-term matrix for topic modeling

        Returns:
        --------
        int: Optimal number of topics
        """
        search_params = {'n_components': [2, 3, 5, 7, 9, 13, 15]}

        # Initialize LDA model
        model = LatentDirichletAllocation(learning_method='online',
                                          max_iter=100,
                                          random_state=0)

        # Perform grid search
        gridsearch = GridSearchCV(model,
                                  param_grid=search_params, error_score='raise',
                                  verbose=4)
        gridsearch.fit(cv_matrix)

        # Extract best model and parameters
        best_lda = gridsearch.best_estimator_
        print("[RESULT] Best Model Parameters: ", gridsearch.best_params_)
        print("[RESULT] Best Log Likelihood Score: ", gridsearch.best_score_)
        print('[RESULT] Best Model Perplexity: ', best_lda.perplexity(cv_matrix))

        best_topic_number = gridsearch.best_params_['n_components']
        print(f'[FINAL] Optimal topic number: {best_topic_number}')

        return best_topic_number

    @staticmethod
    def get_reduce_dim_on_embedding(embeddings):
        """
        Apply dimensionality reduction to embeddings using PCA

        Parameters:
        -----------
        embeddings : numpy array, high-dimensional embeddings

        Returns:
        --------
        tuple: (reduced_embeddings, number_of_components)
        """
        print('[STATUS] Standardizing embeddings for dimensionality reduction...')
        scaler_train_semantic = StandardScaler()
        scaler_train_semantic.fit(embeddings)

        import joblib
        joblib.dump(scaler_train_semantic, 'scaler_model_from_train_for_pca.pkl')
        scaled_embedding_train = scaler_train_semantic.transform(embeddings)

        print(f'[DEBUG] Embedding matrix type: {type(embeddings)}')
        print('[STATUS] Performing PCA dimensionality reduction...')

        # Evaluate different component numbers for PCA
        k_values_pca = [5, 10, 15, 20, 25, 30, 35, 40]
        best_k_pca = None
        best_variance = 0.0

        for k in k_values_pca:
            pca = PCA(n_components=k)
            pca_result = pca.fit_transform(scaled_embedding_train)
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"[EVAL] K={k}: explained variance = {explained_variance:.4f}")

            if explained_variance > best_variance:
                best_k_pca = k
                best_variance = explained_variance

        print(f'[RESULT] Optimal PCA components: {best_k_pca} (variance: {best_variance:.4f})')

        # Apply final PCA transformation
        import joblib
        pca_train = PCA(n_components=best_k_pca)
        pca_train.fit(scaled_embedding_train)
        joblib.dump(pca_train, 'pca_model_from_train.pkl')
        pca_reduced_embeddings = pca_train.transform(scaled_embedding_train)
        n_components_ica = best_k_pca

        reduced_embeddings = pca_reduced_embeddings
        return reduced_embeddings, n_components_ica

    @staticmethod
    def get_lda_topics(best_topic_number, cv_matrix):
        """
        Perform LDA topic modeling with specified number of topics

        Parameters:
        -----------
        best_topic_number : int, number of topics to extract
        cv_matrix : scipy.sparse matrix, document-term matrix

        Returns:
        --------
        numpy array: Document-topic distribution matrix
        """
        lda = LatentDirichletAllocation(
            n_components=best_topic_number,
            learning_method='online',
            max_iter=100,
            random_state=0
        )
        doc_topic_matrix = lda.fit_transform(cv_matrix)

        return doc_topic_matrix

    @staticmethod
    def calculation_entropy(topic_distribution):
        """
        Calculate entropy of topic distribution

        Parameters:
        -----------
        topic_distribution : list or array, probability distribution over topics

        Returns:
        --------
        float: Entropy value of the distribution
        """
        topic_distribution_cupy = cp.array(topic_distribution, dtype=cp.float32)
        non_zero_probs = topic_distribution_cupy[topic_distribution_cupy > 0]
        entropy = cp.sum(-non_zero_probs * cp.log2(non_zero_probs))

        return float(entropy)

    @staticmethod
    def get_sentiment_lst_dataframe(df_features):
        """
        Perform sentiment analysis on text data with batch processing

        Parameters:
        -----------
        df_features : DataFrame, contains text data for sentiment analysis

        Returns:
        --------
        list: Sentiment labels for each text entry
        """
        batch_size = 512

        # Extract unique texts to avoid redundant processing
        unique_texts = df_features['processed_EventTemplate'].unique().tolist()
        print(f"[INFO] Analyzing {len(unique_texts)} unique texts from {len(df_features)} total rows")

        # Initialize sentiment analysis pipeline
        sentiment_pipeline = pipeline("sentiment-analysis",
                                      model="siebert/sentiment-roberta-large-english",
                                      device=-1)  # CPU execution

        # Prepare dataset for batch processing
        dataset = hdp.from_dict({"text": unique_texts})

        # Batch processing function
        def analyze_batch(batch):
            results = sentiment_pipeline(batch["text"])
            labels = [result['label'] for result in results]
            return {"label": labels}

        # Execute sentiment analysis
        results = dataset.map(analyze_batch, batched=True, batch_size=batch_size)
        unique_labels = results["label"]

        # Map results back to original dataframe
        sentiment_map = dict(zip(unique_texts, unique_labels))
        lst_sentiment_label = df_features['processed_EventTemplate'].map(sentiment_map).tolist()

        return lst_sentiment_label

    def features_extracted_different_features(self, best_topic_number,
                                              doc_topic_df_path,
                                              pre_final_global_features_pkl_path):
        """
        Extract additional features including temporal, statistical, and entropy features

        Parameters:
        -----------
        best_topic_number : int, number of topics for feature naming
        doc_topic_df_path : str, path to topic modeling results
        pre_final_global_features_pkl_path : str, path to save enhanced features
        """
        # Load topic modeling results
        df_feature_full_dataset_all = pd.read_pickle(doc_topic_df_path)
        df_feature_full_dataset_all.info()
        df_feature_full_dataset_all['processed_EventTemplate'] = df_feature_full_dataset_all[
            'processed_EventTemplate'].astype(str)

        # Remove unnecessary columns
        df_feature_full_dataset = df_feature_full_dataset_all.drop(columns=['Date', 'Time'])
        gdf_feature_full_dataset = df_feature_full_dataset

        # ==================== TEMPORAL FEATURE EXTRACTION ====================
        print('[STATUS] Extracting temporal features from timestamps...')
        gdf_feature_full_dataset['Timestamp'] = pd.to_datetime(gdf_feature_full_dataset['Timestamp'])

        # Extract comprehensive temporal features
        temporal_components = ['year', 'month', 'day', 'hour', 'minute', 'second']
        for component in temporal_components:
            gdf_feature_full_dataset[component] = getattr(gdf_feature_full_dataset['Timestamp'].dt, component)

        # Convert node IDs to categorical for efficient processing
        gdf_feature_full_dataset['Node_block_id'] = gdf_feature_full_dataset['Node_block_id'].astype('category')

        # ==================== TEXT STATISTICAL FEATURES ======================
        print('[STATUS] Calculating text statistical features...')
        gdf_feature_full_dataset['num_words'] = gdf_feature_full_dataset[
            'processed_EventTemplate'].str.split().str.len()

        gdf_feature_full_dataset['Character_Count'] = gdf_feature_full_dataset['processed_EventTemplate'].str.len()

        # Remove intermediate topic distribution column
        gdf_feature_full_dataset = gdf_feature_full_dataset.drop('list_topics_distribution', axis=1)

        # ==================== TOPIC ENTROPY CALCULATION ======================
        print('[STATUS] Calculating topic distribution entropy...')
        list_topic = [f"T_{i}" for i in range(best_topic_number)]
        print(f"[INFO] Topic columns: {list_topic}")

        # Calculate entropy for unique topic distributions
        unique_topic_lists = gdf_feature_full_dataset[list_topic].drop_duplicates()
        unique_topic_lists['entropy'] = unique_topic_lists.apply(
            lambda row: FeaturesExtractor.calculation_entropy(row.values.tolist()),
            axis=1
        )

        # Create mapping for efficient entropy assignment
        entropy_map = {
            tuple(row[list_topic]): row['entropy'] for _, row in unique_topic_lists.iterrows()
        }

        # Apply entropy mapping to full dataset
        gdf_feature_full_dataset['entropy'] = [
            entropy_map[tuple(row)] for row in gdf_feature_full_dataset[list_topic].itertuples(index=False, name=None)
        ]

        # Final data preparation and validation
        df_feature_full_dataset = gdf_feature_full_dataset
        print(f'[DEBUG] Final dataset type: {type(df_feature_full_dataset)}')
        print('[DEBUG] Data types:')
        print(df_feature_full_dataset.dtypes)

        # Save enhanced feature set
        df_feature_full_dataset.info()
        df_feature_full_dataset.to_pickle(pre_final_global_features_pkl_path)

        # Display sample results
        cols_to_show = list_topic + ['entropy']
        print('[DEBUG] Sample topic distributions with entropy:')
        print(df_feature_full_dataset[cols_to_show].head(20))

        print('[SUCCESS] Additional feature extraction completed')

    def start_sentiment_extracting(self, df_features, sentiment_df_path):
        """
        Execute sentiment analysis pipeline and save results

        Parameters:
        -----------
        df_features : DataFrame, input data for sentiment analysis
        sentiment_df_path : str, path to save sentiment results
        """
        print('[STATUS] Starting sentiment analysis pipeline...')
        lst_label_sentiment = self.get_sentiment_lst_dataframe(df_features)

        # Validate data integrity
        if len(lst_label_sentiment) != len(df_features):
            print('[ERROR] Data length mismatch in sentiment analysis')
            exit()

        # Store sentiment results
        df_features['sentiment_label'] = list(lst_label_sentiment)
        df_features.to_pickle(sentiment_df_path)
        df_features.info()
        print('[SUCCESS] Sentiment analysis completed and saved')
