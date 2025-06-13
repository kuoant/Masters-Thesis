#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, 
                            confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_TOKENS = 1000
OUTPUT_SEQUENCE_LENGTH = 20
EMBED_DIM = 32
NUM_HEADS = 2
NUM_TRANSFORMER_BLOCKS = 2

#====================================================================================================================
# Data Preprocessing Module
#====================================================================================================================
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
        risky_sample = risky_pool.sample(frac=1, random_state=RANDOM_SEED)
        
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

#====================================================================================================================
# Model Building Module
#====================================================================================================================
class TransformerBlock(layers.Layer):
    """Transformer block implementation"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class TabTransformerModel:
    """TabTransformer model implementation"""
    @staticmethod
    def build_model(cat_features_info, num_numerical):
        """Build the full TabTransformer model"""
        # Input layers
        categorical_inputs = layers.Input(shape=(len(cat_features_info),), name='categorical_inputs')
        numerical_inputs = layers.Input(shape=(num_numerical,), name='numerical_inputs')
        text_inputs = layers.Input(shape=(OUTPUT_SEQUENCE_LENGTH,), name='text_inputs')
        
        # Categorical processing
        embedded_cats = []
        for i, (card, dim) in enumerate(cat_features_info):
            # Create embedding layer for each categorical feature
            emb = layers.Embedding(input_dim=card, output_dim=dim)(categorical_inputs[:, i:i+1])
            embedded_cats.append(emb)
        
        # Stack embeddings along a new axis
        x_cat = layers.Concatenate(axis=1)(embedded_cats)
        
        # Transformer blocks
        for _ in range(NUM_TRANSFORMER_BLOCKS):
            x_cat = TransformerBlock(EMBED_DIM, NUM_HEADS)(x_cat)
        
        # Flatten
        x_cat = layers.Flatten()(x_cat)
        
        # Numerical features
        x_num = layers.Dense(32, activation='relu')(numerical_inputs)
        
        # Text processing
        x_text = layers.Embedding(input_dim=MAX_TOKENS, output_dim=32)(text_inputs)
        x_text = TransformerBlock(32, 2)(x_text)
        x_text = layers.GlobalAveragePooling1D()(x_text)
        
        # Combine features
        x = layers.Concatenate()([x_cat, x_num, x_text])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(
            inputs=[categorical_inputs, numerical_inputs, text_inputs],
            outputs=outputs
        )
        
        return model

#====================================================================================================================
# Training and Evaluation Module
#====================================================================================================================
class ModelTrainer:
    @staticmethod
    def train_model(model, X_train, y_train):
        """Train the TabTransformer model"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        ]
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            (X_train['categorical'], X_train['numerical'], X_train['text']),
            y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks
        )
        
        return model, history

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """Evaluate the model and print metrics"""
        y_pred_proba = model.predict((X_test['categorical'], X_test['numerical'], X_test['text']))
        y_pred = (y_pred_proba.flatten() > 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print("Evaluation Results:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='darkblue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return acc, auc, f1
    
    @staticmethod
    def evaluate_xgboost(model, X_train, y_train, X_test, y_test, use_embeddings=True):
        """Fixed XGBoost evaluation with proper key handling"""
        if use_embeddings:
            # Create input mapping from our data structure to model's expected names
            train_inputs = {
                'categorical_inputs': X_train['categorical'],
                'numerical_inputs': X_train['numerical'],
                'text_inputs': X_train['text']
            }
            
            test_inputs = {
                'categorical_inputs': X_test['categorical'],
                'numerical_inputs': X_test['numerical'],
                'text_inputs': X_test['text']
            }

            # Create embedding model
            embedding_model = Model(
                inputs=model.inputs,
                outputs=model.layers[-2].output
            )
            
            # Get embeddings using properly mapped inputs
            X_train_emb = embedding_model.predict(train_inputs)
            X_test_emb = embedding_model.predict(test_inputs)
            
            features = X_train_emb
            test_features = X_test_emb
            title = "XGBoost on Transformer Embeddings"
        else:
            # Use raw features
            features = np.hstack([X_train['categorical'].numpy(), 
                                X_train['numerical'].numpy()])
            test_features = np.hstack([X_test['categorical'].numpy(), 
                                    X_test['numerical'].numpy()])
            title = "XGBoost on Raw Features"
        
        # Train and evaluate XGBoost
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, 
                                eval_metric='logloss', 
                                random_state=RANDOM_SEED)
        xgb_clf.fit(features, y_train.numpy())
        
        y_pred = xgb_clf.predict(test_features)
        acc = accuracy_score(y_test.numpy(), y_pred)
        cm = confusion_matrix(y_test.numpy(), y_pred)
        
        ModelEvaluator.plot_confusion_matrix(cm, f"XGBoost Confusion Matrix with Embeddings: {use_embeddings}")

        print(f"{title} Accuracy: {acc:.4f}")
        return acc
    
    @staticmethod
    def plot_confusion_matrix(cm, title):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
    


#====================================================================================================================
# Main Execution
#====================================================================================================================
if __name__ == "__main__":
    # 1. Data Preprocessing
    preprocessor = TabularDataPreprocessor()
    df = preprocessor.load_and_prepare_data("data/df_small_sampled.csv")
    X_train, X_test, y_train, y_test, cat_features_info, num_numerical = preprocessor.preprocess_data(df)
    
    # 2. Model Building
    model_builder = TabTransformerModel()
    model = model_builder.build_model(cat_features_info, num_numerical)
    
    # 3. Model Training
    trainer = ModelTrainer()
    trained_model, history = trainer.train_model(model, X_train, y_train)
    
    # 4. Model Evaluation
    evaluator = ModelEvaluator()
    evaluator.evaluate_model(trained_model, X_test, y_test)
    
    # 5. XGBoost Evaluation
    print("\nEvaluating XGBoost on different feature sets:")
    evaluator.evaluate_xgboost(trained_model, X_train, y_train, X_test, y_test, use_embeddings=True)
    evaluator.evaluate_xgboost(trained_model, X_train, y_train, X_test, y_test, use_embeddings=False)


# %%
