import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    f1_score, 
    confusion_matrix,
    average_precision_score,
    classification_report
)
import lightgbm as lgb
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        pass

    def _clean_names(self, X):
        """
        Cleans feature names to avoid characters that LightGBM and other tools might reject.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
            
        X = X.copy()
        X.columns = [
            str(col).replace('[', '').replace(']', '')
            .replace('<', '').replace('>', '')
            .replace('{', '').replace('}', '')
            .replace(',', '').replace(':', '')
            .replace('"', '').replace("'", "")
            .replace(' ', '_')
            for col in X.columns
        ]
        return X

    def validate_data(self, X, y=None, required_columns=None):
        """
        Validates the input data.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        
        if y is not None:
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise ValueError("y must be a pandas Series or numpy array.")
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")
        
        if required_columns:
            missing = [col for col in required_columns if col not in X.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        return True

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluates the model and prints AUC-PR, F1-Score, and Confusion Matrix.
        """
        try:
            self.validate_data(X_test, y_test)
            X_test_clean = self._clean_names(X_test)
            
            y_pred = model.predict(X_test_clean)
            
            # Handle cases where model might not have predict_proba
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_clean)[:, 1]
                precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
                auc_pr = auc(recall, precision)
            else:
                auc_pr = average_precision_score(y_test, y_pred)
                
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"\n--- {model_name} Evaluation ---")
            logger.info(f"AUC-PR: {auc_pr:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info("\nConfusion Matrix:")
            logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
            logger.info("\nClassification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")
            
            return {"Model": model_name, "AUC-PR": auc_pr, "F1-Score": f1}
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise

    def train_logistic_regression(self, X_train, y_train, max_iter=1000, **kwargs):
        """
        Trains a Logistic Regression model.
        """
        try:
            self.validate_data(X_train, y_train)
            X_train_clean = self._clean_names(X_train)
            model = LogisticRegression(max_iter=max_iter, **kwargs)
            model.fit(X_train_clean, y_train)
            return model
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
            raise

    def train_lightgbm(self, X_train, y_train, **kwargs):
        """
        Trains a LightGBM model. Cleans feature names to avoid JSON errors.
        """
        try:
            self.validate_data(X_train, y_train)
            X_train_clean = self._clean_names(X_train)
            
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
            model.fit(X_train_clean, y_train)
            return model
        except Exception as e:
            logger.error(f"Error training LightGBM: {e}")
            raise

    def tune_hyperparameters(self, model, param_grid, X_train, y_train, scoring='average_precision', cv=5):
        """
        Performs GridSearchCV to find the best hyperparameters.
        """
        try:
            self.validate_data(X_train, y_train)
            X_train_clean = self._clean_names(X_train)
            
            logger.info(f"Starting Hyperparameter Tuning for {type(model).__name__}...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train_clean, y_train)
            
            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"Best Score ({scoring}): {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, grid_search.best_params_
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise

    def perform_cross_validation(self, X, y, model, name="Model", k=5):
        """
        Performs Stratified K-Fold cross-validation and reports mean/std.
        """
        try:
            self.validate_data(X, y)
            X_clean = self._clean_names(X)
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            scoring = {'f1': 'f1', 'auc_pr': 'average_precision'}
            
            cv_results = cross_validate(model, X_clean, y, cv=skf, scoring=scoring)
            
            logger.info(f"\n--- {k}-Fold CV Results for {name} ---")
            logger.info(f"F1: {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
            logger.info(f"AUC-PR: {cv_results['test_auc_pr'].mean():.4f} (+/- {cv_results['test_auc_pr'].std():.4f})")
            
            return cv_results
        except Exception as e:
            logger.error(f"Error during cross-validation: {e}")
            raise

    def select_best_model(self, X_train, y_train, X_test, y_test):
        """
        Compares Baseline (LR) and LightGBM, considering performance and interpretability.
        """
        logger.info("\n--- Starting Model Selection Routine ---")
        
        # 1. Baseline: Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train)
        lr_eval = self.evaluate_model(lr_model, X_test, y_test, "Logistic Regression (Baseline)")
        
        # 2. Advanced: LightGBM
        lgb_model = self.train_lightgbm(X_train, y_train)
        lgb_eval = self.evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        
        results = [lr_eval, lgb_eval]
        comparison = self.compare_models(results)
        
        logger.info("\n--- Model Comparison Summary ---")
        logger.info(f"\n{comparison}")
        
        # Logic for selection
        best_model_name = comparison.iloc[0]['Model']
        
        logger.info(f"\nRecommendation: The best model based on AUC-PR is {best_model_name}.")
        logger.info("Note: Logistic Regression (Baseline) offers high interpretability (coefficients).")
        logger.info("      LightGBM offers higher complex pattern recognition but is a 'black box' (requires SHAP/FI).")
        
        if best_model_name == "LightGBM":
            return lgb_model, comparison
        else:
            return lr_model, comparison

    def compare_models(self, results):
        """
        Creates a comparison table from a list of result dictionaries.
        """
        df = pd.DataFrame(results)
        return df.sort_values(by="AUC-PR", ascending=False)

    def save_model(self, model, filepath):
        """
        Saves the trained model to a specified filepath.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model, filepath)
            logger.info(f"Model saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}")
            raise

    def plot_feature_importance(self, model, feature_names, top_n=20):
        """
        Plots feature importance for LightGBM models.
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.show()
        else:
            logger.warning("Model does not have feature_importances_ attribute.")
