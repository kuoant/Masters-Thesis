#====================================================================================================================
# Imports and Constants
#====================================================================================================================
#%%

# Core Libraries
import pandas as pd
import numpy as np

# BERT
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Machine Learning & Preprocessing
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#====================================================================================================================
# Data Preprocessing Module
#====================================================================================================================

# Load and prepare data
df = pd.read_csv("data/twitter.csv", nrows=500, header=None)
df.columns = ['tweet_id', 'game_name', 'sentiment', 'text']
df = df.dropna()
df['text'] = df['text'].str.lower()

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])
class_names = le.classes_

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Hugging Face dataset
train_ds = Dataset.from_pandas(train_df[['text', 'label']])
test_ds = Dataset.from_pandas(test_df[['text', 'label']])

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

#====================================================================================================================
# BERT Module
#====================================================================================================================

# Load pretrained BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(class_names))

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert-results",
    eval_strategy="epoch", 
    save_strategy="no",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_strategy="epoch",  
    logging_steps=10,         
    report_to="all",          
    seed=42,
)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": (preds == labels).mean(),
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

# Train BERT
trainer.train()

#====================================================================================================================
# Evaluation
#====================================================================================================================

# Evaluate
preds = trainer.predict(test_ds)
y_true = test_df['label'].values
y_pred = np.argmax(preds.predictions, axis=1)
print("BERT Fine-Tuned Evaluation:")
print(classification_report(y_true, y_pred, target_names=class_names))


# %%
