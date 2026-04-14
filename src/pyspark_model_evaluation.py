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

class ModelEvaluation:
    """Class for evaluating model performance and feature importance."""

    @staticmethod

    def  evaluation_pyspark(case1_test_pdf, case2_test_pdf, case1_classification_time,
                            case1_Classification_pred_time, case2_Classification_time, case2_Classification_pred_time):
        # ======================================
        # 5) Classification report
        # ======================================
        print("\n================Case1:  TEST CLASSIFICATION REPORT (HASH-split stable) ================")
        print(classification_report(case1_test_pdf["y"].astype(int), case1_test_pdf["final_pred"], digits=4))
        # print(f"[INFO] best_threshold={best_threshold:.4f}, best_offset={best_offset:.3f}, gate_t={gate_t:.4f}")

        print(f"[INFO] Case1 :  GBT fit time: {case1_classification_time / 60:.6f} minutes")
        print(f"[INFO] Case1 : GBT predict time: {case1_Classification_pred_time / 60:.6f} minutes")

        print("\n================Case2:  TEST CLASSIFICATION REPORT ================")
        print(classification_report(case2_test_pdf["y"].astype(int), case2_test_pdf["final_pred"], digits=4))
        print(f"Case2 : final Model classification  completed in {case2_Classification_time:.6f} minutes")
        print(f"Case2 : final Model predicts  completed in {case2_Classification_pred_time:.6f} minutes")


