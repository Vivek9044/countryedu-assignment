import os
import sys
import subprocess
import time

def run_step(command, description):
    print(f"\n==================================================")
    print(f"[RUN] STEP: {description}")
    print(f"Command: {command}")
    print(f"==================================================")
    
    t0 = time.time()
    result = subprocess.run(command, shell=True)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n[FAIL] ERROR: Step failed! (Return code: {result.returncode})")
        sys.exit(result.returncode)
    else:
        print(f"[SUCCESS] SUCCESS: Step completed in {elapsed:.2f} seconds.")

def main():
    # Use the python executable inside our virtual environment
    python_bin = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(python_bin):
        # Fallback if venv is named differently or on other systems, though we configured Windows
        python_bin = "python"
        print(f"Warning: venv python not found at {python_bin}, using system python.")
        
    print("IT Support Cost Optimization Pipeline Runner")
    print(f"Python Binary: {python_bin}")
    print(f"Current Working Directory: {os.getcwd()}")
    
    t_start = time.time()
    
    # 1. Generate Synthetic Data
    run_step(f"{python_bin} src/data_generator.py", "Synthetic Data Generation (300,000 tickets)")
    
    # 2. Run ETL pipeline
    run_step(f"{python_bin} src/etl_pipeline.py", "ETL / Data Pipeline (Ingestion & Feature Engineering)")
    
    # 3. Train Models
    run_step(f"{python_bin} src/models/cost_model.py", "Train Cost Prediction Model (Regression)")
    run_step(f"{python_bin} src/models/escalation_model.py", "Train Escalation Risk Model (Classification)")
    run_step(f"{python_bin} src/models/satisfaction_model.py", "Train Satisfaction Risk Model (Classification)")
    run_step(f"{python_bin} src/models/allocation_optimizer.py", "Run Engineer Allocation Optimizer (LP Optimization)")
    
    # 4. Generate Business Recommendations
    run_step(f"{python_bin} src/recommendations.py", "Generate Quantified Recommendations Engine")
    
    # 5. Run EDA and Notebook Generation
    run_step(f"{python_bin} src/run_eda.py", "Run Exploratory Data Analysis & Generate Notebook")
    
    # 6. Run Automated Testing Suite
    run_step(f"{python_bin} -m unittest tests/test_pipeline.py", "Automated Testing Suite (unittest)")
    
    t_total = time.time() - t_start
    print(f"\n[DONE] CONGRATULATIONS! Complete pipeline ran successfully in {t_total:.2f} seconds.")
    print("All tests passed, all outputs generated, and models saved.")
    print("You can now run the Streamlit dashboard using:")
    print("streamlit run dashboard/app.py")

if __name__ == '__main__':
    main()
