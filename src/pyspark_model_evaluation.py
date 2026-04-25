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

    def  evaluation_pyspark(case1_test_pdf, case2_test_pdf, case1_classification_time,
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


