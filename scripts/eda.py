import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class FraudImbalanceAnalyzer:
    """
    Class to analyze class imbalance in fraud / loan default datasets.
    """

    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col

    def fraud_percentage(self):
        proportions = self.df[self.target_col].value_counts(normalize=True) * 100
        return proportions.get(1, 0), proportions.get(0, 0)

    def imbalance_ratio(self):
        counts = self.df[self.target_col].value_counts()
        if 0 not in counts or 1 not in counts:
            return None # Handle case where one class is missing
        return counts[0] / counts[1]

    def summary(self):
        fraud_pct, non_fraud_pct = self.fraud_percentage()
        ratio = self.imbalance_ratio()
        print("📊 Class Imbalance Summary")
        print("-" * 35)
        print(f"Fraud / Default (1): {fraud_pct:.2f}%")
        print(f"Non-Fraud / Good (0): {non_fraud_pct:.2f}%")
        if ratio:
            print(f"Imbalance Ratio (Non-Fraud : Fraud) = {ratio:.1f} : 1")

    def plot_distribution(self):
        counts = self.df[self.target_col].value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts.index, y=counts.values)
        plt.xlabel("Class Label")
        plt.ylabel("Number of Samples")
        plt.title("Class Distribution")
        plt.show()

class EDAAnalyzer:
    """
    Class for Exploratory Data Analysis.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def univariate_analysis(self, columns=None):
        """Plot distributions for a list of columns. If None, plots all numeric and top categorical columns."""
        if columns is None:
            # Default to numeric columns and some categorical ones
            columns = self.df.select_dtypes(include=[np.number, 'object']).columns.tolist()
            # Remove high cardinality columns like IDs and timestamps for broad EDA
            columns = [col for col in columns if 'id' not in col.lower() and 'time' not in col.lower()]

        for col in columns:
            plt.figure(figsize=(10, 5))
            if self.df[col].dtype in ['int64', 'float64']:
                sns.histplot(self.df[col], kde=True)
                plt.title(f"Distribution of {col}")
            else:
                top_10 = self.df[col].value_counts().index[:10]
                sns.countplot(y=self.df[col], order=top_10)
                plt.title(f"Top 10 categories for {col}")
            plt.show()

    def bivariate_analysis(self, columns=None, target='class'):
        """Analyze relationships between columns and target."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number, 'object']).columns.tolist()
            columns = [col for col in columns if col != target and 'id' not in col.lower() and 'time' not in col.lower()]

        for col in columns:
            plt.figure(figsize=(10, 5))
            if self.df[col].dtype in ['int64', 'float64']:
                sns.boxplot(x=target, y=col, data=self.df)
                plt.title(f"{col} vs {target}")
            else:
                pd.crosstab(self.df[col], self.df[target], normalize='index').plot(kind='bar', stacked=True)
                plt.title(f"{col} distribution across {target}")
            plt.show()
    
    def analyze_fraud_by_country(self, country_col='country', target='class'):
        """Specific analysis for fraud patterns by country."""
        if country_col in self.df.columns:
            country_fraud = self.df.groupby(country_col)[target].mean().sort_values(ascending=False).head(10)
            plt.figure(figsize=(12, 6))
            country_fraud.plot(kind='bar')
            plt.title("Top 10 Countries by Fraud Rate")
            plt.ylabel("Fraud Rate")
            plt.show()
