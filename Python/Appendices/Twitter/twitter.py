#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning & Preprocessing
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

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

class TwitterDataPreprocessor:
    @staticmethod
    def load_and_prepare_data(filepath, nrows=None):
        """Load untitled Twitter data and add column names"""
        # Load data without header
        df = pd.read_csv(filepath, nrows=nrows, header=None)
        
        # Add column names based on observed structure
        df.columns = ['tweet_id', 'game_name', 'sentiment', 'text']
        
        # Clean data
        df = df.dropna()
        df['text'] = df['text'].str.lower()  # Basic normalization
        
        # Encode sentiment labels
        le = LabelEncoder()
        df['sentiment_label'] = le.fit_transform(df['sentiment'])
        
        return df, le.classes_

    @staticmethod
    def preprocess_data(df):
        """Prepare data for model training"""
        train_data, test_data = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        
        # Text vectorization
        text_vectorizer = layers.TextVectorization(
            max_tokens=MAX_TOKENS,
            output_mode='int',
            output_sequence_length=OUTPUT_SEQUENCE_LENGTH
        )
        text_vectorizer.adapt(train_data['text'].values)
        
        # Prepare datasets
        X_train = {'text': text_vectorizer(train_data['text'].values)}
        X_test = {'text': text_vectorizer(test_data['text'].values)}
        y_train = tf.convert_to_tensor(train_data['sentiment_label'].values)
        y_test = tf.convert_to_tensor(test_data['sentiment_label'].values)
        
        return X_train, X_test, y_train, y_test


#====================================================================================================================
# Transformer Block
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

class TwitterTransformerModel:
    """Text-only transformer model for Twitter sentiment"""
    @staticmethod
    def build_model(num_classes):
        text_inputs = layers.Input(shape=(OUTPUT_SEQUENCE_LENGTH,), name='text_inputs')
        
        # Text processing pipeline
        x = layers.Embedding(input_dim=MAX_TOKENS, output_dim=EMBED_DIM)(text_inputs)
        
        # Transformer blocks
        for _ in range(NUM_TRANSFORMER_BLOCKS):
            x = TransformerBlock(EMBED_DIM, NUM_HEADS)(x)
        
        # Classification head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        return Model(inputs=text_inputs, outputs=outputs)
    
    @staticmethod
    def visualize_embeddings(embeddings, labels, class_names, title="Embeddings PCA Visualization"):
        """Visualize embeddings with PCA and sanity checks"""
        # Sanity check 1: Check embedding dimensions
        print(f"\nSanity Check - Embedding shape: {embeddings.shape}")
        
        # Sanity check 2: Check for NaN values
        print(f"Sanity Check - NaN values: {np.isnan(embeddings).sum()}")
        
        # Standardize features before PCA
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Reduce to 2D with PCA
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        embeddings_2d = pca.fit_transform(embeddings_scaled)
        
        # Sanity check 3: Check PCA explained variance
        print(f"Sanity Check - PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ticks=range(len(class_names)))
        plt.clim(-0.5, len(class_names)-0.5)
        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        
        # Add class names to colorbar
        cbar = plt.gcf().axes[-1]
        cbar.set_yticklabels(class_names)
        
        plt.show()


#====================================================================================================================
# XGBoost Module
#====================================================================================================================

class BaselineXGBoost:
    @staticmethod
    def prepare_data(df):
        """Prepare data for baseline XGBoost with one-hot encoded text features"""
        train_data, test_data = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        
        # Basic text preprocessing
        train_data['text'] = train_data['text'].str.lower()
        test_data['text'] = test_data['text'].str.lower()
        
        # Create simple character n-gram features (as a simple baseline)
        vectorizer = CountVectorizer(
            analyzer='char', 
            ngram_range=(2, 4),  # character 2-4 grams
            max_features=1000    # limit number of features
        )
        
        X_train_text = vectorizer.fit_transform(train_data['text'])
        X_test_text = vectorizer.transform(test_data['text'])
        
        y_train = train_data['sentiment_label'].values
        y_test = test_data['sentiment_label'].values
        
        return X_train_text, X_test_text, y_train, y_test
    
#====================================================================================================================
# Main Execution
#====================================================================================================================

def main():
    # 1. Load and preprocess data
    df, class_names = TwitterDataPreprocessor.load_and_prepare_data(
        "data/twitter.csv", nrows=500)
    
    # 2. Run baseline XGBoost on raw features
    print("\n=== Baseline XGBoost (Raw Text Features) ===")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = BaselineXGBoost.prepare_data(df)
    
    xgb_baseline = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_names),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=RANDOM_SEED
    )
    xgb_baseline.fit(X_train_raw, y_train_raw)
    
    print("\nBaseline XGBoost Evaluation:")
    baseline_pred = xgb_baseline.predict(X_test_raw)
    print(classification_report(y_test_raw, baseline_pred, target_names=class_names))

    # Get class probabilities
    baseline_proba = xgb_baseline.predict_proba(X_test_raw)

    # Compute multi-class AUC (One-vs-Rest)
    auc_score = roc_auc_score(y_test_raw, baseline_proba, multi_class='ovr')
    print(f"AUC Score (multi-class OVR): {auc_score:.4f}")
        
    # 3. Run transformer + XGBoost pipeline (original code)
    print("\n=== Transformer + XGBoost Pipeline ===")
    X_train, X_test, y_train, y_test = TwitterDataPreprocessor.preprocess_data(df)
    
    # Build and train transformer model
    model = TwitterTransformerModel.build_model(len(class_names))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining transformer model...")
    model.fit(
        X_train['text'],
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate transformer
    print("\nTransformer Model Evaluation:")
    y_pred = model.predict(X_test['text']).argmax(axis=1)
    print(classification_report(y_test.numpy(), y_pred, target_names=class_names))
    
    # Get class probabilities from transformer
    y_proba = model.predict(X_test['text'])

    # Compute multi-class AUC (One-vs-Rest)
    auc_score_tf = roc_auc_score(y_test.numpy(), y_proba, multi_class='ovr')
    print(f"AUC Score (Transformer Only, OVR): {auc_score_tf:.4f}")

    embedding_model = Model(
    inputs=model.input,  # Use model.input instead of model.inputs for single input
    outputs=model.layers[-3].output
    )
    
    print("\nExtracting embeddings...")
    # Convert the input to numpy array to avoid the warning
    X_train_text_np = X_train['text'].numpy() if hasattr(X_train['text'], 'numpy') else X_train['text']
    X_test_text_np = X_test['text'].numpy() if hasattr(X_test['text'], 'numpy') else X_test['text']

    X_train_emb = embedding_model.predict(X_train_text_np)
    X_test_emb = embedding_model.predict(X_test_text_np)

    
    print("\nTraining XGBoost on embeddings...")
    xgb_emb = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_names),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=RANDOM_SEED
    )
    xgb_emb.fit(X_train_emb, y_train.numpy())
    
    print("\nXGBoost with Embeddings Evaluation:")
    emb_pred = xgb_emb.predict(X_test_emb)
    print(classification_report(y_test.numpy(), emb_pred, target_names=class_names))

    
    # Get class probabilities
    emb_proba = xgb_emb.predict_proba(X_test_emb)

    # Compute multi-class AUC (One-vs-Rest)
    auc_score_emb = roc_auc_score(y_test.numpy(), emb_proba, multi_class='ovr')
    print(f"AUC Score (Embeddings + XGBoost, OVR): {auc_score_emb:.4f}")


    # Visualization of embeddings with PCA
    print("\n=== Embedding Visualization ===")

    # Visualize test embeddings using the class method
    TwitterTransformerModel.visualize_embeddings(
        X_test_emb, 
        y_test.numpy(), 
        class_names,
        title="Test Set Embeddings (PCA)"
    )
    
    # For comparison, visualize raw text features (from baseline)
    TwitterTransformerModel.visualize_embeddings(
        X_test_raw.toarray(),  # Convert sparse matrix to dense
        y_test_raw,
        class_names,
        title="Baseline Raw Features (PCA)"
    )

    # 1. Transformer model parameters
    print("\nTransformer Model Summary:")
    model.summary()
    print(f"Total transformer trainable parameters: {model.count_params():,}")

    # 2. Baseline XGBoost
    booster_df_base = xgb_baseline.get_booster().trees_to_dataframe()
    print(f"\nBaseline XGBoost: {booster_df_base['Tree'].nunique()} trees, "
        f"{booster_df_base.shape[0]} nodes, ~{booster_df_base.shape[0] * 3:,} estimated params")

    # 3. XGBoost on embeddings
    booster_df_emb = xgb_emb.get_booster().trees_to_dataframe()
    print(f"\nXGBoost on Embeddings: {booster_df_emb['Tree'].nunique()} trees, "
        f"{booster_df_emb.shape[0]} nodes, ~{booster_df_emb.shape[0] * 3:,} estimated params")

if __name__ == "__main__":
    main()


# %%
