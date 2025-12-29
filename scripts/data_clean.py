import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        pass

    def load_data(self, filepath):
        """Loads data from a CSV file with error handling."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            raise

    def handle_missing_values(self, df):
        """Impute missing values: numeric with median, categorical with 'Unknown'."""
        try:
            df = df.copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            if not categorical_cols.empty:
                df[categorical_cols] = df[categorical_cols].fillna('Unknown')
            logger.info("Handled missing values.")
            return df
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise

    def remove_duplicates(self, df):
        """Removes duplicate rows."""
        return df.drop_duplicates()

    def correct_data_types(self, df):
        """Corrects data types: timestamps to datetime, IP to integer."""
        if 'signup_time' in df.columns:
            df['signup_time'] = pd.to_datetime(df['signup_time'])
        if 'purchase_time' in df.columns:
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        if 'ip_address' in df.columns:
            df['ip_address'] = df['ip_address'].apply(lambda x: int(float(x)))
        return df

    def merge_with_geo(self, fraud_df, ip_df):
        """
        Merges fraud data with country data based on IP range.
        Optimized using pd.merge_asof after sorting.
        """
        try:
            required_fraud = ['ip_address']
            required_ip = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
            
            for col in required_fraud:
                if col not in fraud_df.columns:
                    raise ValueError(f"Fraud data missing required column: {col}")
            for col in required_ip:
                if col not in ip_df.columns:
                    raise ValueError(f"IP data missing required column: {col}")

            # Ensure IP addresses are integers
            fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)
            ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
            ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)

            # Sort for merge_asof
            fraud_df = fraud_df.sort_values('ip_address')
            ip_df = ip_df.sort_values('lower_bound_ip_address')

            # Merge based on lower bound
            merged_df = pd.merge_asof(
                fraud_df, 
                ip_df, 
                left_on='ip_address', 
                right_on='lower_bound_ip_address'
            )

            # Filter out cases where IP is not within the upper bound
            mask = (merged_df['ip_address'] >= merged_df['lower_bound_ip_address']) & \
                   (merged_df['ip_address'] <= merged_df['upper_bound_ip_address'])
            
            merged_df.loc[~mask, 'country'] = 'Unknown'
            
            logger.info("Successfully merged data with geolocation.")
            # Cleanup extra columns from ip_df
            return merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
        except Exception as e:
            logger.error(f"Error merging with geo: {e}")
            raise

    def engineer_features(self, df):
        """Add time-based features and transaction frequency."""
        # Time-based features
        if 'signup_time' in df.columns and 'purchase_time' in df.columns:
            df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.dayofweek
            df['month'] = df['purchase_time'].dt.month

        # Transaction frequency: number of transactions per user
        if 'user_id' in df.columns:
            df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
        
        # Transaction velocity (simulated: transactions per device/IP in a short window would be better, 
        # but here we'll just do device usage frequency as a proxy for now)
        if 'device_id' in df.columns:
            df['device_usage_count'] = df.groupby('device_id')['device_id'].transform('count')

        return df

    def transform_data(self, df, categorical_cols=None, numerical_cols=None):
        """Normalize numerical features and encode categorical ones."""
        # Work on a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        if numerical_cols:
            scaler = StandardScaler()
            # Handle cases where columns might be missing
            existing_num = [col for col in numerical_cols if col in df.columns]
            if existing_num:
                df[existing_num] = scaler.fit_transform(df[existing_num])
        
        if categorical_cols:
            # Simple One-Hot Encoding via get_dummies
            existing_cat = [col for col in categorical_cols if col in df.columns]
            if existing_cat:
                df = pd.get_dummies(df, columns=existing_cat, drop_first=True)
            
        return df

    def prepare_for_modeling(self, df, target_col='class'):
        """
        Final pruning for model training. 
        Drops metadata, encodes remaining categories, and ensures purely numeric output.
        """
        df = df.copy()
        
        # 1. Drop metadata columns that have no predictive value or aren't numeric
        metadata_cols = ['user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address']
        df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
        
        # 2. Encode any remaining categorical columns (e.g., 'country')
        cat_cols = df.select_dtypes(include=['object']).columns
        if not cat_cols.empty:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            
        # 3. Handle boolean columns (convert to int)
        bool_cols = df.select_dtypes(include=['bool']).columns
        if not bool_cols.empty:
            df[bool_cols] = df[bool_cols].astype(int)
        
        # 4. Final check: move target column to the end if it exists
        if target_col in df.columns:
            cols = [col for col in df.columns if col != target_col] + [target_col]
            df = df[cols]
            
        return df