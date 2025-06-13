import tensorflow as tf
from tensorflow.keras import layers, Model
from src.constants import *

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