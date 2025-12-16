import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go


# 1. APP CONFIGURATION
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. LOAD ASSETS (Model & Scaler)
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the Jupyter Notebook to generate 'model.pkl' and 'scaler.pkl'.")
        return None, None

model, scaler = load_model()


# 3. HELPER CONSTANTS (For Visuals & Imputation)

# Medians calculated from the training set (to handle 0 inputs)
MEDIANS = {
    'Glucose': 117.0,
    'BloodPressure': 72.0,
    'SkinThickness': 29.0,
    'Insulin': 125.0,
    'BMI': 32.3
}

# Max values for normalization in Radar Chart (approx from dataset)
MAX_VALS = {
    'Pregnancies': 17, 'Glucose': 200, 'BloodPressure': 122, 
    'SkinThickness': 99, 'Insulin': 846, 'BMI': 67, 
    'DiabetesPedigreeFunction': 2.42, 'Age': 81
}

# Average profiles for comparison (Healthy vs Diabetic)
AVG_HEALTHY = [3.3, 110, 68, 19.6, 68.8, 30.3, 0.43, 31.2]
AVG_DIABETIC = [4.9, 141, 70.8, 22.2, 100.3, 35.1, 0.55, 37.1]


# 4. SIDEBAR - USER INPUTS

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("Patient Data")
    st.write("Enter the patient's clinical metrics below.")
   
    
    st.markdown("---")
    
    # Input groups
    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input("Glucose", 0, 200, 120, help="Plasma glucose concentration")
        bp = st.number_input("Blood Pressure", 0, 140, 70, help="Diastolic blood pressure (mm Hg)")
        insulin = st.number_input("Insulin", 0, 900, 80, help="2-Hour serum insulin (mu U/ml)")
        dpf = st.number_input("Pedigree Func", 0.0, 2.5, 0.5, step=0.01, help="Diabetes Pedigree Function")
        
    with col2:
        skin = st.number_input("Skin Thickness", 0, 100, 20, help="Triceps skin fold thickness (mm)")
        bmi = st.number_input("BMI", 0.0, 70.0, 30.0, help="Body mass index")
        age = st.number_input("Age", 0, 120, 33)
        
    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title("⚕️ HealthGuard Analysis System")
st.markdown("### AI-Powered Diabetes Risk Assessment")

if predict_btn and model:
    # --- A. Data Preprocessing ---
    # 1. Create array (Handle 0s by replacing with Median if necessary, or just warn)
    # Note: In the training step, we imputed 0s. A robust app should do the same.
    user_input = {
        'Pregnancies': pregnancies,
        'Glucose': glucose if glucose > 0 else MEDIANS['Glucose'],
        'BloodPressure': bp if bp > 0 else MEDIANS['BloodPressure'],
        'SkinThickness': skin if skin > 0 else MEDIANS['SkinThickness'],
        'Insulin': insulin if insulin > 0 else MEDIANS['Insulin'],
        'BMI': bmi if bmi > 0 else MEDIANS['BMI'],
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    input_data = np.array([list(user_input.values())])
    
    # 2. Scale Data
    input_scaled = scaler.transform(input_data)
    
    # --- B. Prediction ---
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    # --- C. Results Section ---
    col_res, col_viz = st.columns([1, 2])
    
    with col_res:
        st.markdown("#### Risk Probability")
        
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Diabetes Risk %"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "salmon"}],
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if prediction[0] == 1:
            st.error(f"**High Risk Detected**\nThe model predicts a high probability of diabetes.")
        else:
            st.success(f"**Low Risk Detected**\nThe model predicts a low probability of diabetes.")

    with col_viz:
        st.markdown("#### Patient Profile vs. Population Averages")
        st.info("This chart compares the patient's metrics (Blue) against the average Healthy (Green) and Diabetic (Red) profiles.")
        
        # Prepare data for Radar Chart (Normalize values 0-1 for visualization)
        categories = list(user_input.keys())
        
        # Helper to normalize
        def normalize(values):
            return [v / MAX_VALS[k] for k, v in zip(categories, values)]
        
        user_norm = normalize(list(user_input.values()))
        healthy_norm = normalize(AVG_HEALTHY)
        diabetic_norm = normalize(AVG_DIABETIC)
        
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=healthy_norm, theta=categories, fill='toself', name='Avg Healthy',
            line=dict(color='green', dash='dot')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=diabetic_norm, theta=categories, fill='toself', name='Avg Diabetic',
            line=dict(color='red', dash='dot')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=user_norm, theta=categories, fill='toself', name='Patient Input',
            line=dict(color='blue', width=3)
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

else:
    # Default State
    st.markdown("""
    👈 **Please enter patient details in the sidebar to begin.**
    
    ### How to use this tool:
    1. Input clinical parameters such as Glucose, BMI, and Age.
    2. Click **Predict Risk**.
    3. Analyze the Risk Gauge and Radar Chart to understand the contributing factors.
        
    Note :  leave Pregnancy as zero for male patients.
    """)
    
    # Optional: Show dataset preview if available
    try:
        df = pd.read_csv('diabetes.csv')
        with st.expander("Peek at the Dataset"):
            st.dataframe(df.head())
    except:
        pass