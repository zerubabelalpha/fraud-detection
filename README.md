# Fraud Detection Data Analysis and Engineering

This project focuses on identifying fraudulent activities in e-commerce and credit card transactions. It includes a robust pipeline for data cleaning, geolocation integration, feature engineering, and exploratory data analysis (EDA).

## Project Overview

Fraud detection is a critical challenge in the digital economy. This project provides a structured approach to:
1.  **Clean and Preprocess** transaction data.
2.  **Integrate Geolocation** information using IP address ranges.
3.  **Engineer Features** that capture behavior patterns (transaction frequency, time-based features).
4.  **Analyze Data** to understand fraud patterns and class imbalances.
5.  **Prepare Data** for machine learning models.

## Project Structure

```text
fraud-detection/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
├── data/
│   ├── raw/                # Original datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│   └── processed/          # Cleaned and engineered data
├── notebooks/              # Jupyter notebooks for analysis and modeling
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb      # Modular modeling pipeline
│   ├── shap-explainability.ipynb
│   └── README.md
├── scripts/                # Python scripts for modular functionality
│   ├── data_clean.py       # DataCleaner class for preprocessing
│   ├── eda.py              # EDAAnalyzer classes
│   ├── imbalance_handler.py # ImbalanceHandler class for SMOTE
│   └── modeling_utils.py   # ModelTrainer class for specialized modeling
├── tests/                  # Unit test suite
│   ├── test_modeling.py    # Tests for ModelTrainer
│   └── __init__.py
├── requirements.txt        # Project dependencies
└── README.md               # Main project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Core Modules

### 1. Data Cleaning (`scripts/data_clean.py`)
Provides the `DataCleaner` class to handle:
- Missing value imputation (numeric: median, categorical: 'Unknown').
- Duplicate removal.
- Data type correction (timestamps and IP addresses).
- Range-based IP to country merging.
- Feature scaling and encoding.

### 2. Exploratory Data Analysis (`scripts/eda.py`)
Provides the `EDAAnalyzer` class to:
- Visualize univariate and bivariate distributions.
- Analyze class imbalance ratios.
- Identify fraud patterns by country.

### 3. Modular Modeling (`scripts/modeling_utils.py`)
Provides the `ModelTrainer` class for a streamlined modeling pipeline:
- **Training**: Standardized helpers for Logistic Regression and LightGBM.
- **Evaluation**: Unified reporting for AUC-PR, F1-Score, and Confusion Matrices.
- **Cross-Validation**: 5-fold Stratified K-Fold for reliable performance estimation.
- **Comparison**: Automatic summary table generation for side-by-side analysis.

### 4. Imbalance Handling (`scripts/imbalance_handler.py`)
Provides the `ImbalanceHandler` class to handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

## Usage

### Using Scripts
Run the full processing pipeline:
```bash
python scripts/process_data.py
```

### Using in Notebooks
You can import the modules directly into your notebooks:
```python
from scripts.data_clean import DataCleaner
from scripts.modeling_utils import ModelTrainer

cleaner = DataCleaner()
trainer = ModelTrainer()
# ... analysis and modeling logic ...
```

## Testing & CI/CD

### Running Tests
The project uses `unittest` for automated verification of the core modules.
```bash
python -m unittest discover tests
```

### GitHub Actions (CI)
A continuous integration pipeline is configured in `.github/workflows/unittests.yml`. It automatically installs dependencies and runs unit tests on every push or pull request to the `master` or `main` branches.

