import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from .constants import *

class ModelEvaluator:
    @staticmethod
    def evaluate_with_gnn(model, data, original_features_np, labels):
        # Get GNN embeddings
        model.eval()
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            hidden_embeddings = model.conv1(x, edge_index)
            hidden_embeddings = F.relu(hidden_embeddings)
            embeddings_np = hidden_embeddings.cpu().numpy()
        
        # Combine features
        combined_features = np.hstack((original_features_np, embeddings_np))
        
        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            combined_features, labels, test_size=TEST_SIZE, 
            random_state=RANDOM_SEED, stratify=labels
        )
        
        # Train and evaluate XGBoost
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        
        print(f"XGBoost Accuracy (with GNN embeddings): {acc:.4f}")
        ModelEvaluator.plot_confusion_matrix(cm, "XGBoost Confusion Matrix (GNN-enhanced features)")
        
        return acc, cm
    
    @staticmethod
    def evaluate_without_gnn(original_features_np, labels):
        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            original_features_np, labels, test_size=TEST_SIZE, 
            random_state=RANDOM_SEED, stratify=labels
        )
        
        # Train and evaluate XGBoost
        xgb_raw = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
        xgb_raw.fit(X_train, y_train)
        y_pred_raw = xgb_raw.predict(X_val)
        acc_raw = accuracy_score(y_val, y_pred_raw)
        cm_raw = confusion_matrix(y_val, y_pred_raw)
        
        print(f"XGBoost Accuracy (original features only): {acc_raw:.4f}")
        ModelEvaluator.plot_confusion_matrix(cm_raw, "Confusion Matrix: XGBoost (Original Features Only)")
        
        return acc_raw, cm_raw
    
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