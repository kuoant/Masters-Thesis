
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from .constants import *  # Import project constants

class DataPreprocessor:
    @staticmethod
    def load_and_sample_data(filepath):
        df = pd.read_csv(filepath)
        
        # Split by class and sample
        df_non_default = df[df['Default'] == 0]
        df_default = df[df['Default'] == 1]
        
        df_non_default_sampled = resample(df_non_default, replace=False, 
                                        n_samples=SAMPLE_SIZES['non_default'], 
                                        random_state=RANDOM_SEED)
        df_default_sampled = resample(df_default, replace=False, 
                                     n_samples=SAMPLE_SIZES['default'], 
                                     random_state=RANDOM_SEED)
        
        # Combine and shuffle
        df_small = pd.concat([df_non_default_sampled, df_default_sampled])
        df_small = df_small.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        df_small = df_small.drop(columns=['LoanID'])
        df_small.to_csv("df_small_sampled.csv", index=False)
        
        return df_small
    
    @staticmethod
    def preprocess_data(df):
        # Encode categorical variables
        label_encoder = LabelEncoder()
        for col in CATEGORICAL_COLS:
            df[col] = label_encoder.fit_transform(df[col])
        
        # Split features and target
        X = df.drop(columns=["Default"])
        y = df["Default"]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        
        # Standardize numerical data
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
        X_test_scaled[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])
        
        return X_train_scaled, X_test_scaled, y_train, y_test
