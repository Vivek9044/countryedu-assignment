import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Global models directory
MODELS_DIR = 'models_saved'
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, 'cost_model.joblib')

def train_and_evaluate_cost_model():
    cleaned_path = 'data/cleaned_tickets.csv'
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data not found at {cleaned_path}. Run ETL pipeline first.")
        
    print(f"Loading cleaned dataset for cost prediction model...")
    df = pd.read_csv(cleaned_path)
    
    # Define features and target
    # We predict cost using features known at creation time
    categorical_features = ['issue_category', 'priority', 'client_type', 'engineer_experience_level']
    numerical_features = ['is_weekend', 'hour_of_day']
    target = 'total_cost'
    
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
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }
    
    best_r2 = -float('inf')
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
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'TrainTime': train_time}
        print(f"  {name} Evaluated in {train_time:.2f}s | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_name = name
            
    print(f"\nWinner: {best_name} (R² = {best_r2:.4f})")
    
    # Save the winner
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best cost model to {MODEL_PATH}")
    
    # Save comparison report
    report_path = 'reports/cost_model_report.txt'
    with open(report_path, 'w') as f:
        f.write("--- COST PREDICTION MODEL COMPARISON ---\n")
        for name, metrics in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
            f.write(f"  MAE: {metrics['MAE']:.2f}\n")
            f.write(f"  R2: {metrics['R2']:.4f}\n")
            f.write(f"  Training Time: {metrics['TrainTime']:.2f}s\n\n")
        f.write(f"Selected Model: {best_name}\n")
    print(f"Saved comparison report to {report_path}")
    
    return results

def predict_single_ticket(ticket_dict):
    """
    Predict the total cost for a single ticket.
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
    pred = model.predict(df)[0]
    
    latency = time.time() - t0
    
    # Latency constraint assertion
    assert latency < 5.0, f"Latency violation! Prediction took {latency:.4f} seconds, which exceeds the 5-second limit."
    
    return {
        'prediction': float(pred),
        'latency_seconds': latency
    }

if __name__ == '__main__':
    train_and_evaluate_cost_model()
    
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
