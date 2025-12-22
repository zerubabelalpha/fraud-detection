import os
import pandas as pd
from data_clean import DataCleaner
from eda import EDAAnalyzer, FraudImbalanceAnalyzer

def run_pipeline(fraud_data_path, ip_data_path):
    cleaner = DataCleaner()
    
    print("--- Loading Data ---")
    fraud_df = cleaner.load_data(fraud_data_path)
    ip_df = cleaner.load_data(ip_data_path)
    
    print("--- Cleaning Data ---")
    fraud_df = cleaner.handle_missing_values(fraud_df)
    fraud_df = cleaner.remove_duplicates(fraud_df)
    fraud_df = cleaner.correct_data_types(fraud_df)
    
    print("--- Geolocation Integration ---")
    merged_df = cleaner.merge_with_geo(fraud_df, ip_df)
    
    print("--- Feature Engineering ---")
    merged_df = cleaner.engineer_features(merged_df)
    
    print("--- Class Imbalance Analysis ---")
    analyzer = FraudImbalanceAnalyzer(merged_df, 'class')
    analyzer.summary()
    
    print("--- Exploratory Data Analysis ---")
    eda = EDAAnalyzer(merged_df)
    # Perform some sample analysis
    eda.analyze_fraud_by_country()
    
    print("--- Data Transformation ---")
    # Example scaling and encoding (adjust column names as needed for your specific dataset)
    numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'user_transaction_count']
    categorical_cols = ['source', 'browser', 'sex']
    
    final_df = cleaner.transform_data(merged_df, categorical_cols=categorical_cols, numerical_cols=numerical_cols)
    
    return final_df

if __name__ == "__main__":
    # Update paths to absolute paths or relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fraud_path = os.path.join(base_dir, 'data', 'raw', 'Fraud_Data.csv')
    ip_path = os.path.join(base_dir, 'data', 'raw', 'IpAddress_to_Country.csv')
    
    if os.path.exists(fraud_path) and os.path.exists(ip_path):
        final_processed_df = run_pipeline(fraud_path, ip_path)
        print("Final DataFrame Shape:", final_processed_df.shape)
        # Optionally save the processed data
        # final_processed_df.to_csv(os.path.join(base_dir, 'data', 'processed', 'cleaned_fraud_data.csv'), index=False)
    else:
        print("Data files not found. Please check paths.")
