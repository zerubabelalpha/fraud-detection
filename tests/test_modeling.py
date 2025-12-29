import sys
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
sys.path.append(os.path.abspath('./'))

from scripts.modeling_utils import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = ModelTrainer()
        # Create synthetic data for testing
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.X_train, self.X_test = self.X[:80], self.X[80:]
        self.y_train, self.y_test = self.y[:80], self.y[80:]

    def test_train_logistic_regression(self):
        model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_lightgbm(self):
        model = self.trainer.train_lightgbm(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))

    def test_evaluate_model(self):
        model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
        results = self.trainer.evaluate_model(model, self.X_test, self.y_test, "Test Model")
        self.assertIn("AUC-PR", results)
        self.assertIn("F1-Score", results)
        self.assertEqual(results["Model"], "Test Model")

    def test_compare_models(self):
        res1 = {"Model": "A", "AUC-PR": 0.8, "F1-Score": 0.7}
        res2 = {"Model": "B", "AUC-PR": 0.9, "F1-Score": 0.8}
        comparison = self.trainer.compare_models([res1, res2])
        self.assertEqual(comparison.iloc[0]["Model"], "B")

if __name__ == "__main__":
    unittest.main()
