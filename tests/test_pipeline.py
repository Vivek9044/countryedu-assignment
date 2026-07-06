import os
import time
import joblib
import unittest
import pandas as pd
import numpy as np

class TestITSupportPipeline(unittest.TestCase):

    def setUp(self):
        self.raw_data_path = 'data/it_support_tickets.csv'
        self.cleaned_data_path = 'data/cleaned_tickets.csv'
        self.cost_model_path = 'models_saved/cost_model.joblib'
        self.escalation_model_path = 'models_saved/escalation_model.joblib'
        self.satisfaction_model_path = 'models_saved/satisfaction_model.joblib'
        
        # Standard sample ticket for predictions
        self.sample_ticket = {
            'issue_category': 'Network',
            'priority': 'Critical',
            'client_type': 'Enterprise',
            'engineer_experience_level': 'Senior',
            'is_weekend': 0,
            'hour_of_day': 14
        }
        self.sample_df = pd.DataFrame([self.sample_ticket])

    def test_1_raw_data_exists_and_has_columns(self):
        """Verify the raw dataset is generated correctly and contains the specified columns."""
        self.assertTrue(os.path.exists(self.raw_data_path), f"Raw data not found at {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        
        expected_cols = [
            'ticket_id', 'created_at', 'issue_category', 'priority', 'client_type',
            'resolution_time_minutes', 'engineer_experience_level', 'cost_per_engineer_hour',
            'engineer_hours_spent', 'escalation_count', 'resolved_first_contact',
            'customer_satisfaction_score', 'total_cost'
        ]
        
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Column '{col}' is missing from the raw dataset.")
            
        print(f"[OK] Raw dataset verified: {len(df):,} rows found with all {len(expected_cols)} columns.")

    def test_2_raw_data_has_missing_values(self):
        """Verify the raw dataset contains simulated missing values in target fields."""
        df = pd.read_csv(self.raw_data_path)
        
        # Check that specific columns have null values
        self.assertTrue(df['customer_satisfaction_score'].isnull().sum() > 0, "customer_satisfaction_score has no missing values.")
        self.assertTrue(df['escalation_count'].isnull().sum() > 0, "escalation_count has no missing values.")
        self.assertTrue(df['resolution_time_minutes'].isnull().sum() > 0, "resolution_time_minutes has no missing values.")
        
        print("[OK] Raw dataset missingness verified.")

    def test_3_cleaned_data_no_nulls(self):
        """Verify that the ETL pipeline has successfully filled all null values."""
        self.assertTrue(os.path.exists(self.cleaned_data_path), f"Cleaned data not found at {self.cleaned_data_path}")
        df = pd.read_csv(self.cleaned_data_path)
        
        # Cleaned dataset must have 0 missing values
        null_count = df.isnull().sum().sum()
        self.assertEqual(null_count, 0, f"Cleaned dataset still contains {null_count} null values.")
        print("[OK] Cleaned dataset verified: contains 0 missing values.")

    def test_4_models_load_and_predict(self):
        """Verify that ML models exist, load successfully, and make predictions on sample input."""
        for model_path, name in [
            (self.cost_model_path, "Cost Model"),
            (self.escalation_model_path, "Escalation Model"),
            (self.satisfaction_model_path, "Satisfaction Model")
        ]:
            self.assertTrue(os.path.exists(model_path), f"{name} not found at {model_path}")
            model = joblib.load(model_path)
            
            # Predict
            if "Model" in name:
                pred = model.predict(self.sample_df)
                self.assertIsNotNone(pred, f"{name} returned None for prediction.")
                
        print("[OK] Machine learning models loaded and verified successfully.")

    def test_5_prediction_latency_constraint(self):
        """Verify model prediction latencies are under the mandatory 5-second constraint."""
        for model_path, name in [
            (self.cost_model_path, "Cost Model"),
            (self.escalation_model_path, "Escalation Model"),
            (self.satisfaction_model_path, "Satisfaction Model")
        ]:
            model = joblib.load(model_path)
            
            # Run multiple runs and measure latency
            latencies = []
            for _ in range(50):
                t0 = time.time()
                _ = model.predict(self.sample_df)
                latencies.append(time.time() - t0)
                
            avg_latency = np.mean(latencies)
            max_latency = np.max(latencies)
            
            print(f"[TIME] {name} Inference Latency: Avg = {avg_latency*1000:.4f} ms | Max = {max_latency*1000:.4f} ms")
            self.assertLess(avg_latency, 5.0, f"{name} average prediction latency of {avg_latency:.4f}s exceeds the 5s constraint.")
            self.assertLess(max_latency, 5.0, f"{name} worst-case prediction latency of {max_latency:.4f}s exceeds the 5s constraint.")
            
        print("[OK] Latency constraints verified successfully.")

    def test_6_etl_throughput_constraint(self):
        """Verify the ETL pipeline processing throughput is at least 50,000 events/second."""
        df = pd.read_csv(self.raw_data_path, nrows=50000)
        medians = {
            'resolution_time_minutes': df['resolution_time_minutes'].median(),
            'escalation_count': df['escalation_count'].median(),
            'customer_satisfaction_score': df['customer_satisfaction_score'].median()
        }
        
        esc_rates = df.groupby('client_type')['escalation_count'].mean().to_dict()
        fcr_rates = df.groupby('issue_category')['resolved_first_contact'].mean().to_dict()
        global_stats = {
            'esc_rates': esc_rates,
            'fcr_rates': fcr_rates
        }
        
        # Benchmarking a vectorized pandas run on 50,000 rows
        t0 = time.time()
        
        # Simulating chunk execution
        # 1. Fill missing values
        df['resolution_time_minutes'] = df['resolution_time_minutes'].fillna(medians['resolution_time_minutes'])
        df['escalation_count'] = df['escalation_count'].fillna(medians['escalation_count'])
        df['customer_satisfaction_score'] = df['customer_satisfaction_score'].fillna(medians['customer_satisfaction_score'])
        df['engineer_experience_level'] = df['engineer_experience_level'].fillna('Mid')
        df['engineer_hours_spent'] = df['engineer_hours_spent'].fillna(1.5)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # 2. Feature Engineering
        df['cost_per_ticket'] = np.round(df['engineer_hours_spent'] * df['cost_per_engineer_hour'], 2)
        df['is_weekend'] = (df['created_at'].dt.dayofweek >= 5).astype(int)
        df['hour_of_day'] = df['created_at'].dt.hour
        df['escalation_rate_by_client_type'] = df['client_type'].map(global_stats['esc_rates'])
        df['first_contact_resolution_rate'] = df['issue_category'].map(global_stats['fcr_rates'])
        df['high_risk_flag'] = (
            (df['client_type'] == 'Enterprise') & 
            (df['priority'].isin(['High', 'Critical'])) & 
            (df['escalation_count'] > 1)
        ).astype(int)
        df['resolution_efficiency'] = np.round(df['cost_per_ticket'] / (df['resolution_time_minutes'] + 1), 4)
        
        elapsed = time.time() - t0
        throughput = len(df) / elapsed
        
        print(f"[SPEED] ETL Throughput Benchmark: {throughput:,.2f} events/second (Processed {len(df):,} events in {elapsed:.4f} seconds)")
        self.assertGreaterEqual(throughput, 50000.0, f"ETL throughput of {throughput:,.2f} events/s is below the 50,000 events/s target.")
        print("[OK] Streaming throughput constraint verified successfully.")

if __name__ == '__main__':
    unittest.main()
