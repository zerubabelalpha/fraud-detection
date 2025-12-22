import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.over_sampling import SMOTE

class ImbalanceHandler:
    """
    Handles class imbalance handling using SMOTE.
    """
    def __init__(self, random_state=42):
        self.smote = SMOTE(random_state=random_state)

    def resample_smote(self, X_train, y_train):
        """
        Applies SMOTE to the training data.
        Only apply to training data to avoid leakage!
        """
        print(f"Original shape: {X_train.shape}")
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        print(f"Resampled shape: {X_resampled.shape}")
        return X_resampled, y_resampled

    def plot_comparison(self, y_before, y_after):
        """
        Visualizes class distribution before and after SMOTE.
        """
        plt.figure(figsize=(12, 5))

        # Before Plot
        plt.subplot(1, 2, 1)
        sns.countplot(x=y_before)
        plt.title("Class Distribution (Before SMOTE)")
        plt.xlabel("Class (0: Non-Fraud, 1: Fraud)")
        plt.ylabel("Count")

        # After Plot
        plt.subplot(1, 2, 2)
        sns.countplot(x=y_after)
        plt.title("Class Distribution (After SMOTE)")
        plt.xlabel("Class (0: Non-Fraud, 1: Fraud)")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.show()



