#%%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Replace with your path
os.chdir("/Users/fabiankuonen/Desktop/Masters Thesis/Python/Modular Version/Modular Solution GNN/src")


from src.data_preprocessing import TabularDataPreprocessor
from src.models.transformer import TabTransformerModel
from src.models.trainer import ModelTrainer
from src.evaluation import ModelEvaluator

if __name__ == "__main__":
    # 1. Data Preprocessing
    preprocessor = TabularDataPreprocessor()
    df = preprocessor.load_and_prepare_data("../data/processed/df_small_sampled.csv")
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
