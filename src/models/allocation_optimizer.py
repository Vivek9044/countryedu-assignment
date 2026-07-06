import os
import pandas as pd
import pulp

def run_allocation_optimizer():
    cleaned_path = 'data/cleaned_tickets.csv'
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data not found at {cleaned_path}. Run ETL pipeline first.")
        
    print("Loading cleaned dataset for allocation optimization...")
    df = pd.read_csv(cleaned_path)
    
    # 1. Calculate weekly ticket volume and parameters
    # The dataset spans 1 year (52 weeks)
    num_weeks = 52.0
    
    # Unique values
    categories = df['issue_category'].unique().tolist()
    priorities = df['priority'].unique().tolist()
    levels = ['Junior', 'Mid', 'Senior']
    
    # Hourly rates
    hourly_costs = {'Junior': 35.0, 'Mid': 65.0, 'Senior': 110.0}
    
    # Compute weekly volume for each (category, priority)
    volume_df = df.groupby(['issue_category', 'priority']).size().reset_index(name='annual_volume')
    volume_df['weekly_volume'] = volume_df['annual_volume'] / num_weeks
    volume_dict = volume_df.set_index(['issue_category', 'priority'])['weekly_volume'].to_dict()
    
    # Compute average resolution hours and engineer hours spent for each (category, priority, level)
    # Convert minutes to hours for resolution time
    df['resolution_hours'] = df['resolution_time_minutes'] / 60.0
    
    res_time_df = df.groupby(['issue_category', 'priority', 'engineer_experience_level'])['resolution_hours'].mean().reset_index()
    res_time_dict = res_time_df.set_index(['issue_category', 'priority', 'engineer_experience_level'])['resolution_hours'].to_dict()
    
    eng_hours_df = df.groupby(['issue_category', 'priority', 'engineer_experience_level'])['engineer_hours_spent'].mean().reset_index()
    eng_hours_dict = eng_hours_df.set_index(['issue_category', 'priority', 'engineer_experience_level'])['engineer_hours_spent'].to_dict()
    
    # Fill in any missing combinations with sensible defaults based on category-priority means and level multipliers
    global_mean_res = df['resolution_hours'].mean()
    global_mean_eng = df['engineer_hours_spent'].mean()
    level_multipliers = {'Junior': 1.5, 'Mid': 1.0, 'Senior': 0.7}
    
    for c in categories:
        for p in priorities:
            cat_prio_subset = df[(df['issue_category'] == c) & (df['priority'] == p)]
            mean_res_cp = cat_prio_subset['resolution_hours'].mean() if len(cat_prio_subset) > 0 else global_mean_res
            mean_eng_cp = cat_prio_subset['engineer_hours_spent'].mean() if len(cat_prio_subset) > 0 else global_mean_eng
            for l in levels:
                if (c, p, l) not in res_time_dict:
                    res_time_dict[(c, p, l)] = mean_res_cp * (level_multipliers[l] / level_multipliers['Mid'])
                if (c, p, l) not in eng_hours_dict:
                    eng_hours_dict[(c, p, l)] = mean_eng_cp * (level_multipliers[l] / level_multipliers['Mid'])
                    
    # Define weekly engineer capacity (hours) - set to a large pool to prevent capacity-driven infeasibility
    capacities = {
        'Junior': 500 * 40.0, # 20000 hours
        'Mid': 500 * 40.0,    # 20000 hours
        'Senior': 500 * 40.0   # 20000 hours
    }
    
    # 2. Formulate Linear Programming Problem
    prob = pulp.LpProblem("IT_Support_Engineer_Allocation", pulp.LpMinimize)
    
    # Decision variables: x[c, p, l] = number of tickets of category c, priority p assigned to level l per week
    x = pulp.LpVariable.dicts("x", 
                              ((c, p, l) for c in categories for p in priorities for l in levels), 
                              lowBound=0, 
                              cat='Continuous')
    
    # Objective function: Minimize total weekly cost (based on engineer hours spent, not customer wait time)
    prob += pulp.lpSum(x[c, p, l] * eng_hours_dict[(c, p, l)] * hourly_costs[l] 
                       for c in categories for p in priorities for l in levels)
    
    # Constraints
    # 1. Demand constraint: All weekly tickets must be processed
    for c in categories:
        for p in priorities:
            v_cp = volume_dict.get((c, p), 0)
            prob += pulp.lpSum(x[c, p, l] for l in levels) == v_cp, f"Demand_{c}_{p}"
            
    # 2. Capacity constraint: Total hours spent by each level cannot exceed capacity (based on engineer hours spent)
    for l in levels:
        prob += pulp.lpSum(x[c, p, l] * eng_hours_dict[(c, p, l)] for c in categories for p in priorities) <= capacities[l], f"Capacity_{l}"
        
    # 3. SLA constraint: Average resolution time for category c, priority p must meet the SLA target (based on resolution time)
    for c in categories:
        for p in priorities:
            v_cp = volume_dict.get((c, p), 0)
            if v_cp > 0:
                # Dynamic SLA target:
                if p == 'Critical':
                    achievable_sla = res_time_dict[(c, p, 'Mid')] * 1.10
                elif p == 'High':
                    achievable_sla = res_time_dict[(c, p, 'Mid')] * 1.60
                elif p == 'Medium':
                    achievable_sla = res_time_dict[(c, p, 'Mid')] * 2.20
                else: # Low
                    achievable_sla = res_time_dict[(c, p, 'Mid')] * 3.50
                    
                prob += pulp.lpSum(x[c, p, l] * res_time_dict[(c, p, l)] for l in levels) <= achievable_sla * v_cp, f"SLA_{c}_{p}"
                
    # Solve the optimization problem
    print("Solving LP Allocation Optimization...")
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    print(f"Optimization Status: {pulp.LpStatus[status]}")
    
    # 3. Extract and display results
    opt_allocation = []
    total_weekly_optimized_cost = 0.0
    total_weekly_hours_used = {'Junior': 0.0, 'Mid': 0.0, 'Senior': 0.0}
    
    for c in categories:
        for p in priorities:
            v_cp = volume_dict.get((c, p), 0)
            row = {'category': c, 'priority': p, 'weekly_volume': v_cp}
            
            actual_res_hours_sum = 0.0
            
            for l in levels:
                val = x[c, p, l].varValue
                row[f'allocated_{l.lower()}'] = val
                
                hours_used = val * eng_hours_dict[(c, p, l)]
                total_weekly_hours_used[l] += hours_used
                total_weekly_optimized_cost += hours_used * hourly_costs[l]
                actual_res_hours_sum += val * res_time_dict[(c, p, l)]
                
            row['avg_res_time_hours_opt'] = (actual_res_hours_sum / v_cp) if v_cp > 0 else 0.0
            opt_allocation.append(row)
            
    opt_df = pd.DataFrame(opt_allocation)
    
    # Save optimized allocation table
    os.makedirs('reports', exist_ok=True)
    opt_df.to_csv('reports/optimized_allocation.csv', index=False)
    print("Saved optimized allocation table to reports/optimized_allocation.csv")
    
    # Compare with current cost
    # Current weekly cost in the dataset, scaled to account for null rows
    df['total_cost'] = df['engineer_hours_spent'] * df['cost_per_engineer_hour']
    mean_ticket_cost = df['total_cost'].mean()
    current_weekly_cost = mean_ticket_cost * (len(df) / num_weeks)
    
    weekly_savings = current_weekly_cost - total_weekly_optimized_cost
    annual_savings = weekly_savings * num_weeks
    pct_savings = (weekly_savings / current_weekly_cost) * 100
    
    print("\n--- OPTIMIZATION RESULTS ---")
    print(f"Current Weekly Support Cost: ${current_weekly_cost:,.2f}")
    print(f"Optimized Weekly Support Cost: ${total_weekly_optimized_cost:,.2f}")
    print(f"Weekly Savings: ${weekly_savings:,.2f}")
    print(f"Projected Annual Cost Savings: ${annual_savings:,.2f} ({pct_savings:.2f}% reduction)")
    
    print("\nEngineer Capacity Utilization:")
    for l in levels:
        used = total_weekly_hours_used[l]
        avail = capacities[l]
        pct = (used / avail) * 100
        print(f"  {l} Engineers: {used:.2f} / {avail:.2f} hours used ({pct:.2f}%)")
        
    print("----------------------------")
    
    # Save optimization summary report
    with open('reports/optimization_summary.txt', 'w') as f:
        f.write("--- LP ALLOCATION OPTIMIZATION SUMMARY ---\n")
        f.write(f"Optimization Status: {pulp.LpStatus[status]}\n")
        f.write(f"Current Weekly Support Cost: ${current_weekly_cost:,.2f}\n")
        f.write(f"Optimized Weekly Support Cost: ${total_weekly_optimized_cost:,.2f}\n")
        f.write(f"Weekly Savings: ${weekly_savings:,.2f}\n")
        f.write(f"Projected Annual Cost Savings: ${annual_savings:,.2f} ({pct_savings:.2f}% reduction)\n\n")
        f.write("Capacity Utilization:\n")
        for l in levels:
            used = total_weekly_hours_used[l]
            avail = capacities[l]
            pct = (used / avail) * 100
            f.write(f"  {l} Engineers: {used:.2f} / {avail:.2f} hours used ({pct:.2f}%)\n")
            
    return {
        'current_weekly_cost': current_weekly_cost,
        'optimized_weekly_cost': total_weekly_optimized_cost,
        'weekly_savings': weekly_savings,
        'annual_savings': annual_savings,
        'pct_savings': pct_savings,
        'utilization': {l: total_weekly_hours_used[l] / capacities[l] for l in levels}
    }

if __name__ == '__main__':
    run_allocation_optimizer()
