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
├── data/
│   ├── raw/                # Original datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│   └── processed/          # Cleaned and engineered data
├── notebooks/              # Jupyter notebooks for analysis and modeling
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│    └── __init__.py
├── scripts/                # Python scripts for modular functionality
│   ├── data_clean.py       # DataCleaner class for preprocessing
│   ├── eda.py              # EDAAnalyzer and FraudImbalanceAnalyzer classes
│   └── process_data.py     # Master script to run the processing pipeline
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
Provides the `EDAAnalyzer` and `FraudImbalanceAnalyzer` classes to:
- Visualize univariate and bivariate distributions.
- Analyze class imbalance ratios.
- Identify fraud patterns by country.

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
from scripts.eda import EDAAnalyzer

cleaner = DataCleaner()
# ... analysis logic ...
```

