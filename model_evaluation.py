import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report


class ModelEvaluation:
    """Class for evaluating model performance and feature importance."""

    @staticmethod
    def evaluation(number_components, y_test_truth, y_test_pred, dataset, X_test):
        """
        Evaluate model performance and compute feature importance metrics.

        Parameters:
        number_components (int): Number of principal components used
        y_test_truth (array): Ground truth labels
        y_test_pred (array): Predicted labels
        dataset (str): Dataset identifier for output file naming
        X_test (DataFrame or array): Test features
        """

        # Print classification report
        print("\n StackingClassifier - Classification Report:")
        print(classification_report(y_test_truth, y_test_pred,
                                    target_names=["Class 0", "Class 1"], digits=3))

        # Define feature names
        feature_names = (
                [f"component_{i + 1}" for i in range(number_components)] +
                [
                    # Text features
                    'sentiment',
                    'dominant_topic',
                    'word_count',
                    'character_count',
                    'entropy',

                    # Temporal features
                    'month',
                    'day',
                    'hour',
                    'minute',
                    'second'
                ]
        )

        # Ensure X_test is a DataFrame
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        # Compute Mutual Information scores
        print("Calculating feature importance using Mutual Information...")
        mi_scores = mutual_info_classif(X_test, y_test_truth)

        # Create and sort MI scores DataFrame
        mi_df = pd.DataFrame({
            "Feature": feature_names[:len(mi_scores)],  # Handle potential length mismatch
            "MI_Score": mi_scores
        }).sort_values(by="MI_Score", ascending=False)

        # Save results to CSV
        output_filename = f"{dataset}_evaluation_mi_scores.csv"
        mi_df.to_csv(output_filename, index=False)
        print(f"Mutual Information scores saved to: {output_filename}")

        return mi_df