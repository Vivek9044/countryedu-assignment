import os
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="IT Support Cost Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def check_pipeline_outputs():
    required_files = [
        'data/cleaned_tickets.csv',
        'models_saved/cost_model.joblib',
        'models_saved/escalation_model.joblib',
        'models_saved/satisfaction_model.joblib',
        'reports/optimized_allocation.csv',
        'reports/recommendations.md'
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    return missing

missing_files = check_pipeline_outputs()

if missing_files:
    st.error("⚠️ Pipeline outputs are missing! Please run the complete pipeline first to generate the dataset, models, and optimizer outputs.")
    st.info("Run the following commands in your terminal:")
    st.code("python src/data_generator.py\npython src/etl_pipeline.py\npython src/models/cost_model.py\npython src/models/escalation_model.py\npython src/models/satisfaction_model.py\npython src/models/allocation_optimizer.py\npython src/recommendations.py\npython src/run_eda.py")
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_tickets.csv')
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df

df = load_data()


cost_model = joblib.load('models_saved/cost_model.joblib')
esc_model = joblib.load('models_saved/escalation_model.joblib')
sat_model = joblib.load('models_saved/satisfaction_model.joblib')


opt_alloc_df = pd.read_csv('reports/optimized_allocation.csv')


@st.cache_data
def parse_recommendation_metrics():
    with open('reports/recommendations.md', 'r') as f:
        content = f.read()
    
    import re
    savings_match = re.search(r"Projected Total Savings: \*\*([$0-9,.]+)\s+per year\*\*", content)
    pct_match = re.search(r"Achieved Cost Reduction\*\*: \*\*([0-9.]+)\%\*\*", content)
    
    savings = savings_match.group(1) if savings_match else "$0.00"
    pct = float(pct_match.group(1)) if pct_match else 0.0
    return savings, pct

projected_savings, projected_pct = parse_recommendation_metrics()


st.sidebar.title("🔍 Filter Controls")
st.sidebar.markdown("Use these to filter the overview insights.")

client_types = ['All'] + sorted(df['client_type'].unique().tolist())
selected_client = st.sidebar.selectbox("Client Type", client_types)

priorities = ['All'] + sorted(df['priority'].unique().tolist())
selected_prio = st.sidebar.selectbox("Priority", priorities)

categories = ['All'] + sorted(df['issue_category'].unique().tolist())
selected_cat = st.sidebar.selectbox("Issue Category", categories)


filtered_df = df.copy()
if selected_client != 'All':
    filtered_df = filtered_df[filtered_df['client_type'] == selected_client]
if selected_prio != 'All':
    filtered_df = filtered_df[filtered_df['priority'] == selected_prio]
if selected_cat != 'All':
    filtered_df = filtered_df[filtered_df['issue_category'] == selected_cat]
    

if len(filtered_df) == 0:
    st.warning("⚠️ No tickets match the selected filters. Please adjust your criteria in the sidebar.")
    st.stop()


st.title("📊 IT Support Cost Optimization & AI Copilot")
st.markdown("Optimize support routing, reduce escalations, improve customer satisfaction, and predict ticket metrics in real-time.")


col1, col2, col3, col4 = st.columns(4)

with col1:
    total_cost = filtered_df['total_cost'].sum()
    st.metric(
        label="Total Support Cost (Selected)",
        value=f"${total_cost:,.2f}"
    )

with col2:
    avg_sat = filtered_df['customer_satisfaction_score'].mean()
    st.metric(
        label="Average Satisfaction (CSAT)",
        value=f"{avg_sat:.2f} / 5.0",
        delta=f"{avg_sat - 4.2:.2f} vs Target 4.2"
    )

with col3:
    escalation_rate = (filtered_df['escalation_count'] > 0).mean() * 100
    st.metric(
        label="Escalation Rate",
        value=f"{escalation_rate:.1f}%"
    )

with col4:
    st.metric(
        label="Projected Annual Savings",
        value=projected_savings,
        delta=f"{projected_pct:.2f}% Target (20% Target)",
        delta_color="normal" if projected_pct >= 20.0 else "inverse"
    )


st.markdown("### Cost Optimization Progress")
progress_val = min(1.0, projected_pct / 20.0)
st.progress(progress_val)
st.markdown(f"**Target Achievement Status:** {projected_pct:.2f}% / 20.00% target reached.")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview & EDA", 
    "🤖 Live Prediction Playground", 
    "💡 Quantified Recommendations", 
    "⚙️ Engineer Allocation Optimizer",
    "🧠 Model Insights"
])


with tab1:
    st.subheader("Interactive Insights Explorer")
    st.write("These charts dynamically update in real-time based on the sidebar filters.")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 1. Average Ticket Cost by Category")
        cat_cost = filtered_df.groupby('issue_category')['total_cost'].mean().reset_index()
        fig_cat = px.bar(cat_cost, x='total_cost', y='issue_category', orientation='h',
                         labels={'total_cost': 'Avg Cost ($)', 'issue_category': 'Category'},
                         color='total_cost', color_continuous_scale='Viridis')
        fig_cat.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)
        st.caption("Takeaway: Cloud and Security issues are the costliest categories due to their complexity.")
        
    with col_chart2:
        st.markdown("#### 2. Average Ticket Cost by Priority")
        prio_cost = filtered_df.groupby('priority')['total_cost'].mean().reset_index()
        fig_prio = px.bar(prio_cost, x='priority', y='total_cost',
                          labels={'total_cost': 'Avg Cost ($)', 'priority': 'Priority'},
                          color='total_cost', color_continuous_scale='Magma',
                          category_orders={'priority': ['Low', 'Medium', 'High', 'Critical']})
        fig_prio.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, showlegend=False)
        st.plotly_chart(fig_prio, use_container_width=True)
        st.caption("Takeaway: Critical/High priority issues carry higher costs because they route to more senior staff.")

    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.markdown("#### 3. Escalation Impact on Ticket Cost")
        plot_df = filtered_df.copy()
        plot_df['is_escalated_str'] = (plot_df['escalation_count'] > 0).map({False: 'Not Escalated', True: 'Escalated'})
        fig_esc = px.box(plot_df, x='is_escalated_str', y='total_cost',
                         color='is_escalated_str',
                         labels={'is_escalated_str': 'Escalation Status', 'total_cost': 'Cost ($)'},
                         color_discrete_map={'Not Escalated': '#10b981', 'Escalated': '#f43f5e'})
        fig_esc.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, showlegend=False)
        st.plotly_chart(fig_esc, use_container_width=True)
        st.caption("Takeaway: Escalated tickets cost significantly more on average due to multiple diagnostic cycles.")
        
    with col_chart4:
        st.markdown("#### 4. Customer Satisfaction (CSAT) Distribution")
        sat_counts = filtered_df['customer_satisfaction_score'].value_counts().reset_index()
        sat_counts.columns = ['CSAT Score', 'Count']
        fig_sat = px.bar(sat_counts, x='CSAT Score', y='Count',
                         labels={'CSAT Score': 'CSAT Score (1-5)', 'Count': 'Tickets Count'},
                         color='CSAT Score', color_continuous_scale='RdYlGn')
        fig_sat.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, showlegend=False)
        st.plotly_chart(fig_sat, use_container_width=True)
        st.caption("Takeaway: Average CSAT score must be kept above the 4.2 line (currently: average baseline is 4.38).")
        
    col_chart5, col_chart6 = st.columns(2)
    
    with col_chart5:
        st.markdown("#### 5. First Contact Resolution (FCR) vs CSAT")
        plot_df['fcr_str'] = plot_df['resolved_first_contact'].map({False: 'No (Multiple Bounces)', True: 'Yes (Resolved Instantly)'})
        fcr_sat = plot_df.groupby('fcr_str')['customer_satisfaction_score'].mean().reset_index()
        fig_fcr = px.bar(fcr_sat, x='fcr_str', y='customer_satisfaction_score',
                         labels={'fcr_str': 'Resolved on First Contact', 'customer_satisfaction_score': 'Avg CSAT'},
                         color='customer_satisfaction_score', color_continuous_scale='RdBu')
        fig_fcr.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, showlegend=False)
        st.plotly_chart(fig_fcr, use_container_width=True)
        st.caption("Takeaway: Standardizing workflows to resolve tickets on first contact is critical for high satisfaction.")
        
    with col_chart6:
        st.markdown("#### 6. Average Resolution Time by Engineer Level")
        eng_perf = filtered_df.groupby('engineer_experience_level')['resolution_time_minutes'].mean().reset_index()
        fig_eng = px.bar(eng_perf, x='engineer_experience_level', y='resolution_time_minutes',
                         labels={'engineer_experience_level': 'Engineer Level', 'resolution_time_minutes': 'Avg Resolution Time (mins)'},
                         color='resolution_time_minutes', color_continuous_scale='Blues',
                         category_orders={'engineer_experience_level': ['Junior', 'Mid', 'Senior']})
        fig_eng.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, showlegend=False)
        st.plotly_chart(fig_eng, use_container_width=True)
        st.caption("Takeaway: Mid and Senior engineers resolve tickets much faster than Junior engineers.")


with tab2:
    st.subheader("Predictive Analytics Playground")
    st.write("Input a new ticket's properties below to run predictions. Our models will estimate cost, escalation risk, and satisfaction risk instantly.")
    
   
    with st.form("prediction_form"):
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            cat_input = st.selectbox("Issue Category", df['issue_category'].unique())
            prio_input = st.selectbox("Priority", ['Low', 'Medium', 'High', 'Critical'])
            client_input = st.selectbox("Client Type", ['Small', 'Medium', 'Enterprise'])
            
        with pred_col2:
            eng_input = st.selectbox("Assigned Engineer Level", ['Junior', 'Mid', 'Senior'])
            weekend_input = st.selectbox("Is Weekend?", ["No", "Yes"])
            hour_input = st.slider("Hour of Day", 0, 23, 12)
            
        submit_btn = st.form_submit_button("Run AI Predictions", key="run_predictions_btn")
        
    if submit_btn:
        # Pre-process inputs
        is_wk = 1 if weekend_input == "Yes" else 0
        
        ticket_dict = {
            'issue_category': cat_input,
            'priority': prio_input,
            'client_type': client_input,
            'engineer_experience_level': eng_input,
            'is_weekend': is_wk,
            'hour_of_day': hour_input
        }
        
       
        t_start = time.time()
        
       
        input_df = pd.DataFrame([ticket_dict])
        
     
        cost_pred = cost_model.predict(input_df)[0]
        
        
        esc_prob = esc_model.predict_proba(input_df)[0, 1]
        esc_pred = esc_model.predict(input_df)[0]
        
     
        sat_prob = sat_model.predict_proba(input_df)[0, 1]
        sat_pred = sat_model.predict(input_df)[0]
        
        latency = time.time() - t_start
        
        st.success(f"🤖 Prediction complete in {latency:.4f} seconds! (Wall-clock constraint: < 5.0 seconds)")
        
      
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(
                label="Predicted Total Cost",
                value=f"${cost_pred:,.2f}"
            )
            st.info(f"Basis: Average resolution hour rate & time for {eng_input} engineers solving a {prio_input} {cat_input} issue.")
            
        with res_col2:
            risk_color = "red" if esc_prob > 0.4 else "green"
            st.markdown(f"#### Escalation Risk")
            st.markdown(f"<h3 style='color:{risk_color};'>{esc_prob*100:.1f}% Probability</h3>", unsafe_allow_html=True)
            if esc_pred == 1:
                st.warning("⚠️ High Risk of escalation detected. Route directly to Mid or Senior engineers.")
            else:
                st.success("✅ Low Risk of escalation. Safe for standard queue.")
                
        with res_col3:
            sat_color = "red" if sat_prob > 0.4 else "green"
            st.markdown(f"#### Satisfaction Risk")
            st.markdown(f"<h3 style='color:{sat_color};'>{sat_prob*100:.1f}% Probability</h3>", unsafe_allow_html=True)
            if sat_pred == 1:
                st.warning("⚠️ High risk of customer CSAT falling below 4.2. Prioritize immediate resolution.")
            else:
                st.success("✅ Satisfaction score expected to remain above 4.2.")


with tab3:
    st.subheader("Data-Driven Cost Optimization Strategy")
    st.write("These recommendations are computed directly by running statistical comparisons and machine learning models on the raw ticket data.")
   
    with open('reports/recommendations.md', 'r') as f:
        recs_md = f.read()
    st.markdown(recs_md)


with tab4:
    st.subheader("Optimal Engineer Allocation Table")
    st.write(
        "This table displays the output of the PuLP Linear Programming optimization. "
        "It details the number of tickets per category/priority that should be assigned to Junior, Mid, and Senior engineers "
        "each week to minimize cost while meeting SLA targets."
    )
    

    st.dataframe(opt_alloc_df.style.format({
        'weekly_volume': '{:.1f}',
        'allocated_junior': '{:.1f}',
        'allocated_mid': '{:.1f}',
        'allocated_senior': '{:.1f}',
        'avg_res_time_hours_opt': '{:.2f}'
    }))
    
    st.markdown("### Optimizer Assumptions & Bounds")
    st.markdown(
        "- **Engineer Capacities (Hours/Week)**: Junior: 4,000 hrs | Mid: 3,400 hrs | Senior: 1,800 hrs.\n"
        "- **Engineer Rates ($/Hr)**: Junior: $35.00 | Mid: $65.00 | Senior: $110.00.\n"
        "- **SLA Targets (Max Resolution Time)**: Critical: 1.5 hrs | High: 3.0 hrs | Medium: 6.0 hrs | Low: 12.0 hrs."
    )

with tab5:
    st.subheader("Model Architectures and Comparison Reports")
    
    col_rep1, col_rep2 = st.columns(2)
    
    with col_rep1:
        st.markdown("#### Cost Regressor Model Summary")
        if os.path.exists('reports/cost_model_report.txt'):
            with open('reports/cost_model_report.txt', 'r') as f:
                st.text(f.read())
                
        st.markdown("#### Escalation Classifier Model Summary")
        if os.path.exists('reports/escalation_model_report.txt'):
            with open('reports/escalation_model_report.txt', 'r') as f:
                st.text(f.read())
                
    with col_rep2:
        st.markdown("#### Satisfaction Risk Classifier Model Summary")
        if os.path.exists('reports/satisfaction_model_report.txt'):
            with open('reports/satisfaction_model_report.txt', 'r') as f:
                st.text(f.read())
                
        st.markdown("#### Technical Implementation Notes")
        st.info(
            "1. **Class Imbalance Handling**: Applied scale_pos_weight parameter proportional to the inverse frequency of positive class counts to train robust XGBoost models.\n"
            "2. **Vectorized Pipeline**: Features are scaled and pre-processed in batches via scikit-learn ColumnTransformer pipelines, delivering sub-millisecond inference times."
        )
