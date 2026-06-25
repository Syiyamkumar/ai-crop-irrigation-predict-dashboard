import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
from utils import get_water_depth

# Load Assets
try:
    model = pickle.load(open('crop_model.pkl', 'rb'))
    model_columns = pickle.load(open('model_columns.pkl', 'rb'))
    with open('crop_data.json', 'r') as f:
        crop_kc_map = json.load(f)
except FileNotFoundError:
    st.error("Error: Required model files or JSON not found. Please run model_trainer.py first.")
    st.stop()

# Page Styling
st.set_page_config(page_title="CWR Smart System", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 Precision Irrigation Dashboard")
st.markdown("---")

# UI Layout: Input Section
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Field Info")
        crop_name = st.selectbox("Crop Type", list(crop_kc_map.keys()))
        soil_type = st.selectbox("Soil Type", ["Clay", "Silt", "Sandy", "Loamy"])
        area = st.number_input("Field Area (Hectares)", min_value=0.1, value=1.0)

    with col2:
        st.subheader("Soil & Weather")
        temp = st.slider("Temperature (°C)", 10, 50, 28)
        moisture = st.slider("Soil Moisture (%)", 0, 100, 35)
        humidity = st.slider("Humidity (%)", 0, 100, 55)

    with col3:
        st.subheader("Environment")
        rainfall = st.number_input("Rainfall (mm)", value=0.0)
        sunlight = st.number_input("Sunlight Hours", value=8.0)
        stage = st.selectbox("Growth Stage", ["Sowing", "Vegetative", "Flowering", "Harvest"])

# Calculation Trigger# Calculation Trigger
if st.button("RUN ANALYSIS", use_container_width=True):
    
    # --- STEP 1: PREPARE DATA ---
    raw_data = {
        'Soil_Type': soil_type, 
        'Temperature_C': temp, 
        'Humidity': humidity,
        'Soil_Moisture': moisture, 
        'Rainfall_mm': rainfall, 
        'Crop_Type': crop_name,
        'Crop_Growth_Stage': stage,
        'Sunlight_Hours': sunlight
    }
    
    # --- STEP 2: AI PREDICTION ---
    input_df = pd.get_dummies(pd.DataFrame([raw_data]))
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    pred_label = model.predict(input_df)[0]
    
    # --- STEP 3: NUMERIC CALCULATION ---
    base_mm = get_water_depth(pred_label)
    kc = crop_kc_map.get(crop_name, 1.0)
    final_mm = base_mm * kc
    total_liters = final_mm * 10000 * area

    # --- STEP 4: DISPLAY RESULTS (NOW THE VARIABLES ARE DEFINED) ---
    st.markdown(f"""
        <div class="result-card">
            <h2 style='color: #2e7d32; margin-top:0;'>Recommended Irrigation: {total_liters:,.0f} Liters</h2>
    """, unsafe_allow_html=True)

    # 5. Quick Metrics Row
    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Intensity", pred_label)
    with res_col2:
        st.metric("Depth", f"{final_mm:.2f} mm")
    with res_col3:
        st.metric("Total Volume", f"{total_liters:,.0f} L", delta="Required")