# IT Support Cost Optimization & AI Predictive Router

A complete, end-to-end data analytics and predictive machine learning project to optimize IT support operations, reduce support costs, manage escalations, and improve customer satisfaction. This system is designed for a volume of 2 million support tickets per year, handling high-throughput ingestion (50,000+ events/sec), robust missing-value imputation, and sub-second machine learning predictions.

---

## Results at a Glance

- **Projected Cost Reduction**: **21.68%** annual savings, representing **$8,752,564.49 saved per year** (exceeding the 20% target).
- **Customer Satisfaction (CSAT)**: Baseline score of **4.38 / 5.0** (exceeding the target of 4.2).
- **Streaming Throughput**: Vectorized ETL processes **123,227.68 tickets/second** (exceeding the 50,000/sec target by **146%**).
- **Model Inference Latency**: Cost, escalation, and satisfaction risk models make predictions in under **20 milliseconds** (exceeding the 5-second SLA constraint by **99.6%**).
- **Predictive Accuracy (XGBoost Regressor)**: Achieved $R^2 = 0.7117$ for ticket cost predictions.

---

## Project Structure

```
it-support-cost-optimization/
├── data/
│   ├── it_support_tickets.csv       # Raw synthetic dataset (300,000 rows)
│   └── cleaned_tickets.csv          # Cleaned, imputed, and engineered dataset
├── src/
│   ├── data_generator.py            # Generates synthetic data with correlations
│   ├── etl_pipeline.py              # Ingests, benchmarks, and cleans raw data
│   ├── models/
│   │   ├── cost_model.py            # Regressor to predict ticket total cost
│   │   ├── escalation_model.py      # Classifier to predict escalation probability
│   │   ├── satisfaction_model.py    # Classifier to predict CSAT risk (< 4.2)
│   │   └── allocation_optimizer.py  # LP model using PuLP to optimize assignments
│   ├── recommendations.py           # recommendation engine with quantified impact
│   └── run_eda.py                   # Programmatically builds the Jupyter notebook
├── notebooks/
│   └── eda.ipynb                    # Exploratory Data Analysis notebook
├── dashboard/
│   └── app.py                       # Streamlit web application dashboard
├── models_saved/                    # Holds serialized trained models (.joblib)
├── reports/
│   ├── Final_Report.md              # Business intelligence final report
│   ├── optimized_allocation.csv     # Solver output allocation matrix
│   ├── recommendations.md           # Quantified recommendations report
│   └── figures/                     # EDA charts saved as PNG
├── tests/
│   └── test_pipeline.py             # Automated unit/integration tests
├── requirements.txt                 # Project dependency list
├── run_pipeline.py                  # End-to-end master runner script
└── README.md                        # This setup guide
```

---

## Installation & Setup

Follow these commands in your powershell or terminal to set up the project:

### 1. Create a Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate the Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Running the Project

You can run the entire project end-to-end with a single script, or execute the parts individually:

### Option A: Run the Complete Pipeline (Recommended)
This runs every component in order, trains all models, solves the optimization matrix, and runs the automated tests:
```powershell
python run_pipeline.py
```

### Option B: Run Steps Individually
If you want to run the scripts step-by-step:

1. **Generate Synthetic Data**:
   ```powershell
   python src/data_generator.py
   ```
2. **Execute ETL Clean/Ingest**:
   ```powershell
   python src/etl_pipeline.py
   ```
3. **Train Predictive Models & Optimizer**:
   ```powershell
   python src/models/cost_model.py
   python src/models/escalation_model.py
   python src/models/satisfaction_model.py
   python src/models/allocation_optimizer.py
   ```
4. **Compile Recommendations**:
   ```powershell
   python src/recommendations.py
   ```
5. **Run EDA Notebook & Visualizations**:
   ```powershell
   python src/run_eda.py
   ```
6. **Execute Validation Test Suite**:
   ```powershell
   python -m unittest tests/test_pipeline.py
   ```

---

## Launching the Interactive Dashboard

Explore the KPI metrics, live ticket predictions, allocation table, and business recommendations on a web UI. The dashboard is configured with **5 interactive tabs**:

1. **Overview & EDA**: Six dynamic, interactive Plotly figures (Category Costs, Priority Costs, Escalation Boxplots, CSAT Distribution, FCR vs. CSAT, and Resolution Speed by Engineer Level) that update in real-time when filters in the sidebar are changed.
2. **Live Prediction Playground**: A form-verified playground where you can enter new ticket configurations to estimate resolution cost, escalation risk, and satisfaction risk with sub-20ms latency.
3. **Quantified Recommendations**: A checklist tracking target achievement (exceeding the 20% cost-reduction goal with a 21.68% cumulative impact).
4. **Engineer Allocation Optimizer**: Displays the PuLP Linear Programming allocation matrix detailing optimal engineer assignments per week.
5. **Model Insights**: Shows validation summaries, RMSE/MAE/F1 metrics, and class imbalance architectures for our XGBoost models.

To run the dashboard, enter this command in your terminal:
```powershell
streamlit run dashboard/app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Core Analytics Decisions

### Why Median/Mode Imputation for Streaming?
During ETL, we compared **Median/Mode Imputation** (0.006s execution time) against **KNN Imputation** (0.362s execution time). While KNN yielded slightly lower reconstruction error, its $O(N^2)$ computation scale is too slow for high-frequency streaming. We chose Median/Mode imputation to meet the **50,000 events/sec streaming constraint**, processing data at over **120,000 events/sec** while keeping predictions stable.
