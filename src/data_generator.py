import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(num_rows=300000, seed=42):
    print(f"Generating {num_rows} synthetic IT support tickets...")
    np.random.seed(seed)
    
    # 1. Ticket IDs
    ticket_ids = [f"TKT-{i:06d}" for i in range(1, num_rows + 1)]
    
    # 2. Created at timestamps (spread over a 1-year period: 2025)
    start_date = datetime(2025, 1, 1)
    # Generate random seconds offset up to 365 days
    seconds_in_year = 365 * 24 * 60 * 60
    random_offsets = np.random.randint(0, seconds_in_year, size=num_rows)
    created_ats = [start_date + timedelta(seconds=int(offset)) for offset in random_offsets]
    created_ats.sort() # Sort chronologically
    
    # 3. Issue categories & priorities & client types
    categories = ['Network', 'Hardware', 'Software', 'Access-Login', 'Email', 'Security', 'Cloud', 'Other']
    cat_probs = [0.15, 0.15, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05]
    issue_categories = np.random.choice(categories, size=num_rows, p=cat_probs)
    
    priorities = ['Low', 'Medium', 'High', 'Critical']
    prio_probs = [0.40, 0.35, 0.18, 0.07]
    ticket_priorities = np.random.choice(priorities, size=num_rows, p=prio_probs)
    
    clients = ['Small', 'Medium', 'Enterprise']
    client_probs = [0.50, 0.35, 0.15]
    client_types = np.random.choice(clients, size=num_rows, p=client_probs)
    
    # Pre-allocate arrays
    engineer_levels = []
    cost_per_hour = []
    hours_spent = []
    escalation_counts = []
    first_contact = []
    satisfaction_scores = []
    
    # Mapping base values for issue category complexity
    category_complexity = {
        'Network': 1.8, 'Hardware': 1.2, 'Software': 1.0, 
        'Access-Login': 0.6, 'Email': 0.5, 'Security': 2.5, 
        'Cloud': 2.0, 'Other': 1.0
    }
    
    print("Simulating correlations...")
    # Loop to simulate realistic correlations
    for i in range(num_rows):
        prio = ticket_priorities[i]
        client = client_types[i]
        cat = issue_categories[i]
        created = created_ats[i]
        
        # Determine engineer level assigned (correlated with priority)
        # Critical/High tickets get Mid/Senior engineers more often
        if prio == 'Critical':
            level = np.random.choice(['Junior', 'Mid', 'Senior'], p=[0.05, 0.25, 0.70])
        elif prio == 'High':
            level = np.random.choice(['Junior', 'Mid', 'Senior'], p=[0.15, 0.45, 0.40])
        elif prio == 'Medium':
            level = np.random.choice(['Junior', 'Mid', 'Senior'], p=[0.40, 0.45, 0.15])
        else: # Low
            level = np.random.choice(['Junior', 'Mid', 'Senior'], p=[0.65, 0.30, 0.05])
        
        engineer_levels.append(level)
        
        # Determine cost per engineer hour based on level
        if level == 'Junior':
            cph = np.random.normal(35, 3)
        elif level == 'Mid':
            cph = np.random.normal(65, 5)
        else: # Senior
            cph = np.random.normal(110, 10)
        cost_per_hour.append(round(cph, 2))
        
        # Base hours spent depends on category complexity and noise
        base_h = category_complexity[cat] * np.random.lognormal(mean=0.3, sigma=0.4)
        
        # Priority multipliers
        prio_mult = {'Low': 0.7, 'Medium': 1.0, 'High': 1.8, 'Critical': 3.0}
        base_h *= prio_mult[prio]
        
        # Experience efficiency multiplier (Seniors are faster, Juniors are slower)
        exp_mult = {'Junior': 1.5, 'Mid': 1.0, 'Senior': 0.7}
        base_h *= exp_mult[level]
        
        # Ensure a minimum amount of work
        base_h = max(0.2, base_h)
        hours_spent.append(round(base_h, 2))
        
        # Escalation count: correlated with client_type, priority, and engineer level
        # Enterprise clients escalate more, critical priority escalated more, junior engineers get escalated more
        esc_lambda = 0.05
        if client == 'Enterprise':
            esc_lambda += 0.25
        elif client == 'Medium':
            esc_lambda += 0.10
            
        if prio == 'Critical':
            esc_lambda += 0.30
        elif prio == 'High':
            esc_lambda += 0.15
            
        if level == 'Junior':
            esc_lambda += 0.20
            
        esc_cnt = np.random.poisson(esc_lambda)
        # Cap escalations at 5
        esc_cnt = min(5, esc_cnt)
        escalation_counts.append(esc_cnt)
        
        # Resolved first contact (FCR): less likely if escalated, or if many hours spent
        if esc_cnt > 0:
            fcr = False
        else:
            # If hours is low, high chance of FCR
            fcr_prob = max(0.05, 0.90 - (base_h * 0.15))
            fcr = np.random.choice([True, False], p=[fcr_prob, 1 - fcr_prob])
        first_contact.append(fcr)
        
        # Customer satisfaction score (1-5):
        # Base rating: 4.4
        csat = 4.4
        # Adjustments
        if level == 'Senior':
            csat += 0.3
        elif level == 'Junior':
            csat -= 0.5
            
        if esc_cnt > 0:
            csat -= (0.4 * esc_cnt)
            
        # Resolution time effect: if hours spent > 5 hours
        if base_h > 5:
            csat -= 0.3
            
        # Weekend or night effect
        is_weekend = created.weekday() >= 5
        is_night = created.hour < 8 or created.hour > 18
        if is_weekend:
            csat -= 0.2
        if is_night:
            csat -= 0.15
            
        # Add random individual noise
        csat += np.random.normal(0, 0.4)
        
        # Clip and round to integer
        csat_score = int(np.clip(round(csat), 1, 5))
        satisfaction_scores.append(csat_score)
        
    # Convert lists to DataFrame
    df = pd.DataFrame({
        'ticket_id': ticket_ids,
        'created_at': created_ats,
        'issue_category': issue_categories,
        'priority': ticket_priorities,
        'client_type': client_types,
        'engineer_experience_level': engineer_levels,
        'cost_per_engineer_hour': cost_per_hour,
        'engineer_hours_spent': hours_spent,
        'escalation_count': escalation_counts,
        'resolved_first_contact': first_contact,
        'customer_satisfaction_score': satisfaction_scores
    })
    
    # Calculate resolution_time_minutes: roughly engineer hours * 60 plus administrative delay
    # Administrative delay is higher for escalated tickets
    admin_delay = df['escalation_count'] * 45 + np.random.exponential(30, size=num_rows)
    df['resolution_time_minutes'] = np.round(df['engineer_hours_spent'] * 60 + admin_delay).astype(int)
    
    # Calculate total_cost
    df['total_cost'] = np.round(df['engineer_hours_spent'] * df['cost_per_engineer_hour'], 2)
    
    # Reorder columns as requested
    df = df[[
        'ticket_id', 'created_at', 'issue_category', 'priority', 'client_type',
        'resolution_time_minutes', 'engineer_experience_level', 'cost_per_engineer_hour',
        'engineer_hours_spent', 'escalation_count', 'resolved_first_contact',
        'customer_satisfaction_score', 'total_cost'
    ]]
    
    # 4. Inject Missing Data (up to 20% concentrated in customer_satisfaction_score, escalation_count, and resolution_time_minutes)
    # We will null out:
    # 20% of customer_satisfaction_score
    # 15% of escalation_count
    # 10% of resolution_time_minutes
    # and maybe 2% in engineer_experience_level and 2% in engineer_hours_spent to show pipeline handles it
    print("Injecting missing values...")
    null_rates = {
        'customer_satisfaction_score': 0.20,
        'escalation_count': 0.15,
        'resolution_time_minutes': 0.10,
        'engineer_experience_level': 0.02,
        'engineer_hours_spent': 0.02
    }
    
    for col, rate in null_rates.items():
        mask = np.random.random(size=num_rows) < rate
        if col in ['customer_satisfaction_score', 'escalation_count', 'resolution_time_minutes']:
            df.loc[mask, col] = np.nan
        elif col == 'engineer_experience_level':
            df.loc[mask, col] = None
        else: # float
            df.loc[mask, col] = np.nan
            
    # Save directory
    os.makedirs('data', exist_ok=True)
    output_path = 'data/it_support_tickets.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path}")
    
    # Summary Statistics
    print("\n--- DATA GENERATION SUMMARY ---")
    print(f"Total Rows: {len(df):,}")
    print("\nMissing values per column:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    for col, pct in missing_pct.items():
        print(f"  {col}: {pct:.2f}% missing")
        
    print("\nBasic distribution statistics:")
    print(df.describe(include=[np.number]).round(2))
    print("\nCategorical distributions:")
    for col in ['issue_category', 'priority', 'client_type', 'engineer_experience_level']:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(dropna=False))
        
    print("---------------------------------")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
