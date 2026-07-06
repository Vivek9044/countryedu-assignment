import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

# Global models directory
MODELS_DIR = 'models_saved'
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, 'satisfaction_model.joblib')

def train_and_evaluate_satisfaction_model():
    cleaned_path = 'data/cleaned_tickets.csv'
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data not found at {cleaned_path}. Run ETL pipeline first.")
        
    print(f"Loading cleaned dataset for satisfaction risk model...")
    df = pd.read_csv(cleaned_path)
    
    # Target variable: satisfaction risk is 1 if score < 4.2, else 0
    # Note: customer_satisfaction_score is between 1 and 5.
    df['satisfaction_risk'] = (df['customer_satisfaction_score'] < 4.2).astype(int)
    
    # Check class distribution
    class_counts = df['satisfaction_risk'].value_counts()
    print(f"Satisfaction Risk class distribution:\n{class_counts}")
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"Imbalance ratio (Low Risk / High Risk): {imbalance_ratio:.2f}")
    
    # Define features and target
    categorical_features = ['issue_category', 'priority', 'client_type', 'engineer_experience_level']
    numerical_features = ['is_weekend', 'hour_of_day']
    target = 'satisfaction_risk'
    
    X = df[categorical_features + numerical_features]
    y = df[target]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])
        
    # Define models
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=imbalance_ratio,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }
    
    best_f1 = -float('inf')
    best_model = None
    best_name = ""
    results = {}
    
    # Train and evaluate each model
    for name, pipeline in models.items():
        print(f"Training {name} model...")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Precision': precision, 
            'Recall': recall, 
            'F1': f1, 
            'ROC_AUC': roc_auc, 
            'ConfusionMatrix': cm,
            'TrainTime': train_time
        }
        print(f"  {name} Evaluated in {train_time:.2f}s | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_name = name
            
    print(f"\nWinner: {best_name} (F1-score = {best_f1:.4f})")
    
    # Save the winner
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best satisfaction risk model to {MODEL_PATH}")
    
    # Save comparison report
    report_path = 'reports/satisfaction_model_report.txt'
    with open(report_path, 'w') as f:
        f.write("--- SATISFACTION RISK MODEL COMPARISON ---\n")
        for name, metrics in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"  Precision: {metrics['Precision']:.4f}\n")
            f.write(f"  Recall: {metrics['Recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['F1']:.4f}\n")
            f.write(f"  ROC AUC: {metrics['ROC_AUC']:.4f}\n")
            f.write(f"  Confusion Matrix:\n    {metrics['ConfusionMatrix'].tolist()}\n")
            f.write(f"  Training Time: {metrics['TrainTime']:.2f}s\n\n")
        f.write(f"Selected Model: {best_name}\n")
    print(f"Saved comparison report to {report_path}")
    
    return results

def predict_single_ticket(ticket_dict):
    """
    Predict satisfaction risk for a single ticket.
    Inputs: ticket_dict (dict) containing:
        - issue_category
        - priority
        - client_type
        - engineer_experience_level
        - is_weekend (0/1)
        - hour_of_day (0-23)
    """
    t0 = time.time()
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        
    model = joblib.load(MODEL_PATH)
    
    # Convert single ticket dict to DataFrame
    df = pd.DataFrame([ticket_dict])
    
    # Run prediction
    prob = model.predict_proba(df)[0, 1]
    pred = model.predict(df)[0]
    
    latency = time.time() - t0
    
    # Latency constraint assertion
    assert latency < 5.0, f"Latency violation! Prediction took {latency:.4f} seconds, which exceeds the 5-second limit."
    
    return {
        'satisfaction_risk_probability': float(prob),
        'satisfaction_risk_prediction': int(pred),
        'latency_seconds': latency
    }

if __name__ == '__main__':
    train_and_evaluate_satisfaction_model()
    
    # Test single prediction
    test_ticket = {
        'issue_category': 'Network',
        'priority': 'Critical',
        'client_type': 'Enterprise',
        'engineer_experience_level': 'Senior',
        'is_weekend': 0,
        'hour_of_day': 14
    }
    result = predict_single_ticket(test_ticket)
    print(f"\nSample Prediction Test: {result}")
