#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_LENGTH = 64  # Max length for BERT input
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

class BERTDataPreprocessor:
    @staticmethod
    def load_and_prepare_data(filepath, nrows=None):
        """Load Twitter data and prepare for BERT"""
        # Load data without header
        df = pd.read_csv(filepath, nrows=nrows, header=None)
        
        # Add column names
        df.columns = ['tweet_id', 'game_name', 'sentiment', 'text']
        
        # Clean data
        df = df.dropna()
        df['text'] = df['text'].str.strip()  # Remove extra whitespace
        
        # Encode sentiment labels
        le = LabelEncoder()
        df['sentiment_label'] = le.fit_transform(df['sentiment'])
        
        return df, le.classes_

    @staticmethod
    def convert_to_input_examples(df, subset='train'):
        """Convert dataframe to InputExample objects for BERT"""
        examples = []
        for i, row in df.iterrows():
            examples.append(
                InputExample(
                    guid=f"{subset}-{i}",
                    text_a=row['text'],
                    label=row['sentiment_label']
                )
            )
        return examples

class BERTModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def convert_examples_to_features(self, examples, max_length=MAX_LENGTH):
        """Convert InputExamples to InputFeatures for BERT"""
        features = []
        
        for ex in examples:
            input_dict = self.tokenizer.encode_plus(
                ex.text_a,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            
            features.append(
                InputFeatures(
                    input_ids=input_dict['input_ids'],
                    attention_mask=input_dict['attention_mask'],
                    token_type_ids=input_dict['token_type_ids'],
                    label=ex.label
                )
            )
        
        def gen():
            for f in features:
                yield (
                    {
                        'input_ids': f.input_ids,
                        'attention_mask': f.attention_mask,
                        'token_type_ids': f.token_type_ids
                    },
                    f.label
                )
        
        return tf.data.Dataset.from_generator(
            gen,
            ({
                'input_ids': tf.int32,
                'attention_mask': tf.int32,
                'token_type_ids': tf.int32
            }, tf.int64),
            ({
                'input_ids': tf.TensorShape([MAX_LENGTH]),
                'attention_mask': tf.TensorShape([MAX_LENGTH]),
                'token_type_ids': tf.TensorShape([MAX_LENGTH])
            }, tf.TensorShape([]))
        )

    def train(self, train_dataset, val_dataset, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
        """Fine-tune BERT model"""
        # Prepare training
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        
        # Train
        history = self.model.fit(
            train_dataset.shuffle(100).batch(BATCH_SIZE),
            validation_data=val_dataset.batch(BATCH_SIZE),
            epochs=epochs,
            batch_size=BATCH_SIZE
        )
        
        return history
    
    def evaluate(self, test_dataset):
        """Evaluate model on test set"""
        results = self.model.evaluate(test_dataset.batch(BATCH_SIZE))
        return results
    
    def predict(self, text):
        """Make predictions on new text"""
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        outputs = self.model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=1)
        predicted_class = tf.argmax(probabilities, axis=1)
        
        return predicted_class.numpy(), probabilities.numpy()

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Load and preprocess data
    df, class_names = BERTDataPreprocessor.load_and_prepare_data(
        "data/twitter.csv", nrows=500)
    
    # 2. Split data
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_SEED)  # 80/10/10 split
    
    # 3. Convert to BERT input format
    train_examples = BERTDataPreprocessor.convert_to_input_examples(train_df, 'train')
    val_examples = BERTDataPreprocessor.convert_to_input_examples(val_df, 'val')
    test_examples = BERTDataPreprocessor.convert_to_input_examples(test_df, 'test')
    
    # 4. Initialize BERT model
    bert_model = BERTModel(num_labels=len(class_names))
    
    # 5. Convert examples to features
    train_dataset = bert_model.convert_examples_to_features(train_examples)
    val_dataset = bert_model.convert_examples_to_features(val_examples)
    test_dataset = bert_model.convert_examples_to_features(test_examples)
    
    # 6. Train the model
    print("\n=== Fine-tuning BERT ===")
    history = bert_model.train(train_dataset, val_dataset)
    
    # 7. Plot training history
    plot_training_history(history)
    
    # 8. Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    test_results = bert_model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results[1]:.4f}")
    
    # 9. Generate classification report
    print("\n=== Classification Report ===")
    y_true = [ex.label for ex in test_examples]
    y_pred = []
    
    for ex in test_examples:
        pred, _ = bert_model.predict(ex.text_a)
        y_pred.append(pred[0])
    
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()
# %%
