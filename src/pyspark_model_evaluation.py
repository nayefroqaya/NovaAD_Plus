import pandas as pd
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.linalg import Vectors
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from pyspark.sql.functions import col

import warnings
import colorama
from pyspark.sql import functions as F

from pyspark.sql.functions import col, when, lit, udf
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col, when, lit
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import time
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.functions import vector_to_array
# ✅ Alias Spark ML classes to avoid ANY shadowing / UnboundLocalError
from pyspark.ml.classification import (
    LogisticRegression as SparkLogisticRegression,
    RandomForestClassifier as SparkRandomForestClassifier,
    GBTClassifier as SparkGBTClassifier,
)
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score


class ModelEvaluation:
    """Class for evaluating model performance and feature importance."""

    @staticmethod

    def  evaluation_pyspark(bert_component, model_case1, model_case2,case1_test_pdf, case2_test_pdf, case1_classification_time,
                            case1_Classification_pred_time, case2_Classification_time, case2_Classification_pred_time):
        # ======================================
        # 5) Classification report
        # ======================================
        # --------------------------
        # FINAL DECISION: case1 vs case2
        # --------------------------

        y1 = case1_test_pdf["y"].astype(int).values
        p1 = case1_test_pdf["final_pred"].astype(int).values

        y2 = case2_test_pdf["y"].astype(int).values
        p2 = case2_test_pdf["final_pred"].astype(int).values

        case1_metrics = {"f1_class1": f1_score(y1, p1, pos_label=1, zero_division=0),
            "recall_class1": recall_score(y1, p1, pos_label=1, zero_division=0),
            "precision_class1": precision_score(y1, p1, pos_label=1, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y1, p1), "fit_time": case1_classification_time,
            "pred_time": case1_Classification_pred_time}

        case2_metrics = {"f1_class1": f1_score(y2, p2, pos_label=1, zero_division=0),
            "recall_class1": recall_score(y2, p2, pos_label=1, zero_division=0),
            "precision_class1": precision_score(y2, p2, pos_label=1, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y2, p2), "fit_time": case2_Classification_time * 60,
            "pred_time": case2_Classification_pred_time * 60}

        print("\n================ FINAL CASE COMPARISON ================")
        print("Case1:", case1_metrics)
        print("Case2:", case2_metrics)

        # Main decision rule: choose better anomaly F1
        if case1_metrics["f1_class1"] > case2_metrics["f1_class1"]:
            final_case = "case1"
        elif case2_metrics["f1_class1"] > case1_metrics["f1_class1"]:
            final_case = "case2"
        else:
            # Tie-breaker 1: higher recall for anomalies
            if case1_metrics["recall_class1"] > case2_metrics["recall_class1"]:
                final_case = "case1"
            elif case2_metrics["recall_class1"] > case1_metrics["recall_class1"]:
                final_case = "case2"
            else:
                # Tie-breaker 2: faster prediction
                final_case = "case1" if case1_metrics["pred_time"] <= case2_metrics["pred_time"] else "case2"


        # -------features importance
        # ============================================================
        # FINAL FEATURE IMPORTANCE
        # Output:
        # - importance of each original feature
        # - one semantic feature score = average importance of 70 BERT components
        # ============================================================

        import numpy as np
        import pandas as pd

        # --------------------------
        # 1) Select final model
        # --------------------------
        if final_case == "case1":
            final_model = model_case1
        else:
            final_model = model_case2

        # --------------------------
        # 2) Feature order inside features_vec_final
        # IMPORTANT:
        # This must match the exact order used when creating features_vec_final
        # --------------------------
        feature_columns = ["sentiment_label_indexed", "Dominant_Topic", "num_words", "Character_Count", "entropy",
            "month", "day", "hour", "minute", "second"]

        bert_component = 70

        # Vector layout:
        # index 0-9   = original features
        # index 10-79 = semantic/BERT reduced components
        n_base = len(feature_columns)
        semantic_start = n_base
        semantic_end = n_base + bert_component

        # --------------------------
        # 3) Get XGBoost importance
        # --------------------------
        booster = final_model.get_booster()

        # You can also use: "weight", "cover", "total_gain"
        importance = booster.get_score(importance_type="gain")

        # Convert XGBoost names f0, f1, ... to numeric indexes
        idx_gain = {}

        for k, v in importance.items():
            idx = int(k.replace("f", ""))
            idx_gain[idx] = float(v)

        # --------------------------
        # 4) Original feature importance
        # --------------------------
        rows = []

        for i, feature_name in enumerate(feature_columns):
            rows.append({"feature": feature_name, "importance_gain": idx_gain.get(i, 0.0)})

        # --------------------------
        # 5) Semantic feature importance
        # Average importance over 70 BERT components
        # --------------------------
        semantic_gains = [idx_gain.get(i, 0.0) for i in range(semantic_start, semantic_end)]

        semantic_avg_gain = float(np.mean(semantic_gains))

        rows.append({"feature": "semantic_feature_avg_70_components", "importance_gain": semantic_avg_gain})

        # --------------------------
        # 6) Final importance table
        # --------------------------
        feature_importance_df = pd.DataFrame(rows)

        feature_importance_df = feature_importance_df.sort_values(by="importance_gain", ascending=False).reset_index(
            drop=True)







        print("\n================Case1:  TEST CLASSIFICATION REPORT (HASH-split stable) ================")
        print(classification_report(case1_test_pdf["y"].astype(int), case1_test_pdf["final_pred"], digits=4))
        # print(f"[INFO] best_threshold={best_threshold:.4f}, best_offset={best_offset:.3f}, gate_t={gate_t:.4f}")

        print(f"[INFO] Case1 :  GBT fit time: {case1_classification_time / 60:.6f} minutes")
        print(f"[INFO] Case1 : GBT predict time: {case1_Classification_pred_time / 60:.6f} minutes")

        print("\n================Case2:  TEST CLASSIFICATION REPORT ================")
        print(classification_report(case2_test_pdf["y"].astype(int), case2_test_pdf["final_pred"], digits=4))
        print(f"Case2 : final Model classification  completed in {case2_Classification_time:.6f} minutes")
        print(f"Case2 : final Model predicts  completed in {case2_Classification_pred_time:.6f} minutes")

        print("\n================Final decision ================")

        print(f"\n[FINAL DECISION] Use {final_case}")
        print(f"\n================ FEATURE IMPORTANCE FOR {final_case.upper()} ================")

        print(f"\n================ FINAL FEATURE IMPORTANCE FOR {final_case.upper()} ================")
        print(feature_importance_df.to_string(index=False))
        print(f"[INFO] Case1 :  GBT fit time: {case1_classification_time / 60:.6f} minutes")
        print(f"[INFO] Case1 : GBT predict time: {case1_Classification_pred_time / 60:.6f} minutes")
        print(f"Case2 : final Model classification  completed in {case2_Classification_time:.6f} minutes")
        print(f"Case2 : final Model predicts  completed in {case2_Classification_pred_time:.6f} minutes")
        with open("output.txt", "w") as f:
            f.write(feature_importance_df.to_string(index=False) + "\n")
            f.write(f"[INFO] Case1 :  GBT fit time: {case1_classification_time / 60:.6f} minutes\n")
            f.write(f"[INFO] Case1 : GBT predict time: {case1_Classification_pred_time / 60:.6f} minutes\n")
            f.write(f"Case2 : final Model classification completed in {case2_Classification_time:.6f} minutes\n")
            f.write(f"Case2 : final Model predicts completed in {case2_Classification_pred_time:.6f} minutes\n")



