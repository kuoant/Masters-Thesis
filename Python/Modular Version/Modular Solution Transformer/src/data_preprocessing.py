import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .constants import *

class TabularDataPreprocessor:
    @staticmethod
    def load_and_prepare_data(filepath):
        """Load and prepare the dataset with job descriptions"""
        df = pd.read_csv(filepath)
        
        # Generate job descriptions
        risky_descriptions = [
            "Worked part-time at a hotel assisting with guest services for 12 months.",
            "Employed part-time in hospitality, primarily at a local hotel front desk for 20 months.",
            "Worked evenings part-time at a hotel restaurant as a server for 10 months."
        ]
        
        generic_descriptions = [
            "Software engineer in a fintech startup. Developed APIs and maintained backend services.",
            "Teacher at a public high school. Responsible for curriculum planning and grading.",
            "Office administrator managing schedules, invoices, and office supplies.",
            "Sales associate at a retail clothing store providing customer support.",
            "Customer service representative at a call center handling billing inquiries.",
            "Freelance content writer producing marketing materials for small businesses.",
            "Warehouse worker managing inventory and handling logistics support.",
            "Data analyst interpreting sales data and creating performance dashboards."
        ]
        
        # Create risky job descriptions for high-risk cases
        risky_pool = df[(df['Default'] == 1) & (df['HasDependents'] == 'Yes')]
        risky_sample = risky_pool.sample(frac=0.7, random_state=RANDOM_SEED)
        
        df['JobDescription'] = None
        for i, idx in enumerate(risky_sample.index):
            df.at[idx, 'JobDescription'] = risky_descriptions[i % len(risky_descriptions)]
        
        # Fill remaining with generic descriptions
        remaining_indices = df[df['JobDescription'].isna()].index
        df.loc[remaining_indices, 'JobDescription'] = np.random.choice(
            generic_descriptions, size=len(remaining_indices))
        
        return df
    
    @staticmethod
    def preprocess_data(df):
        """Split data and prepare features"""
        # Split data
        train_data, test_data = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        
        # Save raw text before encoding
        train_data_raw = train_data.copy()
        test_data_raw = test_data.copy()
        
        # Separate labels
        y_train = train_data['Default']
        y_test = test_data['Default']
        
        # Define columns
        categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus',
                             'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
        
        numerical_columns = train_data.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        numerical_columns = [col for col in numerical_columns 
                            if col not in categorical_columns + ['Default']]
        
        # Encode categorical columns
        for col in categorical_columns:
            train_data[col] = train_data[col].astype('category').cat.codes
            test_data[col] = test_data[col].astype('category').cat.codes
        
        # Get cardinalities
        cat_cardinalities = [train_data[col].nunique() for col in categorical_columns]
        cat_features_info = [(card, EMBED_DIM) for card in cat_cardinalities]
        
        # Scale numerical features
        scaler = StandardScaler()
        train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
        test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
        
        # Prepare text vectorizer
        text_vectorizer = layers.TextVectorization(
            max_tokens=MAX_TOKENS,
            output_mode='int',
            output_sequence_length=OUTPUT_SEQUENCE_LENGTH
        )
        text_vectorizer.adapt(train_data_raw['JobDescription'].fillna('').astype(str).values)
        
        # Vectorize text
        X_train_text = text_vectorizer(train_data_raw['JobDescription'].fillna('').astype(str).values)
        X_test_text = text_vectorizer(test_data_raw['JobDescription'].fillna('').astype(str).values)
        
        # Convert to tensors
        X_train = {
            'categorical': tf.convert_to_tensor(train_data[categorical_columns].values), 
            'numerical': tf.convert_to_tensor(train_data[numerical_columns].values),
            'text': X_train_text
        }
        
        X_test = {
            'categorical': tf.convert_to_tensor(test_data[categorical_columns].values),
            'numerical': tf.convert_to_tensor(test_data[numerical_columns].values),
            'text': X_test_text
        }
        
        y_train = tf.convert_to_tensor(y_train.values)
        y_test = tf.convert_to_tensor(y_test.values)
        
        return X_train, X_test, y_train, y_test, cat_features_info, len(numerical_columns)