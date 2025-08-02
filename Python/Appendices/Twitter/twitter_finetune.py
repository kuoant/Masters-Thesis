#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

# Core Libraries
import pandas as pd
import numpy as np

# BERT & LoRA
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Machine Learning & Preprocessing
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# Handle Warnings
import warnings
warnings.filterwarnings("ignore")

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_LENGTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5

#====================================================================================================================
# Data Preprocessing Module
#====================================================================================================================

def prepare_dataset(df, tokenizer_func):
    dataset = Dataset.from_pandas(df[['text', 'sentiment_label']])
    dataset = dataset.map(tokenizer_func, batched=True)
    dataset = dataset.with_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

class BERTDataPreprocessor:
    @staticmethod
    def load_and_prepare_data(filepath, nrows=None):
        """Load Twitter data and prepare for BERT"""
        df = pd.read_csv(filepath, nrows=nrows, header=None)
        df.columns = ['tweet_id', 'game_name', 'sentiment', 'text']
        df = df.dropna()
        df['text'] = df['text'].str.strip()
        
        le = LabelEncoder()
        df['sentiment_label'] = le.fit_transform(df['sentiment'])
        
        print("\nClass Distribution:")
        print(df['sentiment'].value_counts())
        
        return df, le.classes_

#====================================================================================================================
# BERT Module
#====================================================================================================================

class BERTModelWithLoRA:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        base_model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            return_dict=True
        )
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value", "dense"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            use_dora=True
        )
        
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
    
    def tokenize_data(self, examples):
        tokenized = self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
        tokenized['labels'] = examples['sentiment_label']
        return tokenized

    def train(self, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir='./saved_model_lora',
            eval_strategy='epoch',
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            save_strategy='epoch',
            save_total_limit=2,
            logging_dir='./logs',
            seed=RANDOM_SEED,
            report_to="none",
            remove_unused_columns=False,
            logging_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_accuracy',
            greater_is_better=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        return trainer
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple): 
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': accuracy_score(labels, predictions)}

#====================================================================================================================
# Evaluation Module
#====================================================================================================================

def evaluate_and_report(trainer, test_dataset, class_names):
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Convert logits to probabilities using softmax
    y_score = F.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    
    print(f"\nUnique labels in test set: {np.unique(y_true)}")
    print(f"Expected classes: {class_names}")
    
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
        auc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
        print(f"\nAUC: {auc:.4f}")
    except ValueError as e:
        print(f"\nCould not calculate AUC: {str(e)}")
        if "Only one class present" in str(e):
            print("Warning: AUC requires at least two classes in test set")

#====================================================================================================================
# Main Execution
#====================================================================================================================

def main():
    # 1. Load data
    print("Loading and preparing data...")
    df, class_names = BERTDataPreprocessor.load_and_prepare_data("data/twitter.csv", nrows=500)
    
    # 2. Split data
    print("\nSplitting data...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED)

    # 3. Initialize model
    print("\nInitializing BERT model with LoRA...")
    bert_model = BERTModelWithLoRA(num_labels=len(class_names))

    # 4. Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_df, bert_model.tokenize_data)
    val_dataset = prepare_dataset(val_df, bert_model.tokenize_data)
    test_dataset = prepare_dataset(test_df, bert_model.tokenize_data)

    # 5. Train
    print("\nTraining model...")
    trainer = bert_model.train(train_dataset, val_dataset)

    # 6. Evaluate
    evaluate_and_report(trainer, test_dataset, class_names)

if __name__ == "__main__":
    main()


# %%
