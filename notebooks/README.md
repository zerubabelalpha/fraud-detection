# Notebooks Roadmap

This folder contains Jupyter notebooks that explore the data, engineer features, and evaluate potential models.

## Notebook Descriptions

| Notebook | Description |
| :--- | :--- |
| `eda-fraud-data.ipynb` | Comprehensive EDA on the main e-commerce fraud dataset, including distribution analysis and class imbalance checks. |
| `eda-creditcard.ipynb` | Analysis of the credit card transaction dataset. |
| `feature-engineering.ipynb` | Implementation of advanced features such as transaction velocity, time-since-signup, and geolocation mapping. |
| `modeling.ipynb` | Initial model training and evaluation using cleaned and transformed data. |
| `shap-explainability.ipynb` | Using SHAP to interpret model predictions and understand feature importance. |

## Suggested Execution Order

1.  **Exploratory Data Analysis**: Start with `eda-fraud-data.ipynb` and `eda-creditcard.ipynb` to understand the data.
2.  **Feature Engineering**: Run `feature-engineering.ipynb` to generate the processed datasets used for modeling.
3.  **Modeling**: Use `modeling.ipynb` to train classifiers.
4.  **Explainability**: Finally, use `shap-explainability.ipynb` to explain your model's decisions.

## Prerequisites

Ensure you have the `scripts/` directory in your python path. You can do this by running the notebooks from the project root or by adding the following snippet to the first cell:

```python
import sys
import os
sys.path.append(os.path.abspath('../'))
```

## Data Requirements

These notebooks expect raw data to be present in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`
