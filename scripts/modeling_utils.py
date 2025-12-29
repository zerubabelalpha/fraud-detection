import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    f1_score, 
    confusion_matrix,
    average_precision_score
)
import lightgbm as lgb

class ModelTrainer:
    def __init__(self):
        pass

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluates the model and prints AUC-PR, F1-Score, and Confusion Matrix.
        """
        y_pred = model.predict(X_test)
        
        # Handle cases where model might not have predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            auc_pr = auc(recall, precision)
        else:
            auc_pr = average_precision_score(y_test, y_pred)
            
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n--- {model_name} Evaluation ---")
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return {"Model": model_name, "AUC-PR": auc_pr, "F1-Score": f1}

    def train_logistic_regression(self, X_train, y_train, max_iter=1000, **kwargs):
        """
        Trains a Logistic Regression model.
        """
        model = LogisticRegression(max_iter=max_iter, **kwargs)
        model.fit(X_train, y_train)
        return model

    def train_lightgbm(self, X_train, y_train, **kwargs):
        """
        Trains a LightGBM model.
        """
        params = {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 7,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        params.update(kwargs)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        return model

    def perform_cross_validation(self, X, y, model, name="Model", k=5):
        """
        Performs Stratified K-Fold cross-validation and reports mean/std.
        """
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scoring = {'f1': 'f1', 'auc_pr': 'average_precision'}
        
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
        
        print(f"\n--- {k}-Fold CV Results for {name} ---")
        print(f"F1: {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
        print(f"AUC-PR: {cv_results['test_auc_pr'].mean():.4f} (+/- {cv_results['test_auc_pr'].std():.4f})")
        
        return cv_results

    def compare_models(self, results):
        """
        Creates a comparison table from a list of result dictionaries.
        """
        df = pd.DataFrame(results)
        return df.sort_values(by="AUC-PR", ascending=False)
