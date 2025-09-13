import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
import pickle
import os
from utils import RULModel, preprocess_test_data, predict_rul, COLUMNS

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox {
        background-color: #ffffff;
        border-radius: 5px;
    }
    .stFileUploader {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 5px;
        padding: 10px;
    }
    .result-box {
        background-color: #e9f7ef;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="RUL Prediction Dashboard", page_icon="✈️", layout="wide")

    st.title("✈️ Aircraft Engine RUL Prediction Dashboard")
    st.markdown("""
        A modern tool to predict the **Remaining Useful Life (RUL)** of aircraft engines using the NASA CMAPSS dataset.
        Upload your test data and RUL files, select an engine unit, and visualize the predicted RUL.
    """)

    st.sidebar.header("Input Parameters")
    st.sidebar.markdown("Upload CMAPSS-formatted files and select a unit number.")
    
    test_file = st.sidebar.file_uploader(
        "Upload Test Data (e.g., test_FD001.txt)", 
        type=["txt"],
        help="Upload a space-separated text file with 26 columns (CMAPSS format)."
    )
    rul_file = st.sidebar.file_uploader(
        "Upload RUL Data (e.g., RUL_FD001.txt)", 
        type=["txt"],
        help="Upload the corresponding RUL file with one value per unit."
    )

    if test_file and rul_file:
        with st.spinner("Processing files..."):
            # Save uploaded files temporarily
            test_path = "temp_test.txt"
            rul_path = "temp_rul.txt"
            with open(test_path, "wb") as f:
                f.write(test_file.read())
            with open(rul_path, "wb") as f:
                f.write(rul_file.read())

            # Load scaler and model
            try:
                with open("models/scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                input_dim = 16
                model = RULModel(input_dim=input_dim)
                model.load_state_dict(torch.load("models/rul_ann_model.pth", map_location="cpu"))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
            except FileNotFoundError:
                st.markdown('<div class="error-box">Model or scaler not found. Run the notebook to train and save them.</div>', unsafe_allow_html=True)
                return
            except Exception as e:
                st.markdown(f'<div class="error-box">Error loading model/scaler: {e}</div>', unsafe_allow_html=True)
                return

            # Preprocess data
            try:
                test_data = preprocess_test_data(test_path, rul_path, scaler)
            except Exception as e:
                st.markdown(f'<div class="error-box">Error processing data: {e}</div>', unsafe_allow_html=True)
                return

            # Select unit
            unit_ids = sorted(test_data['unit_number'].unique())
            unit_id = st.sidebar.selectbox("Select Unit Number", unit_ids, help="Choose an engine unit to analyze.")

            col1, col2 = st.columns([1, 2])

            # Process and predict
            with col1:
                st.subheader(f"Results for Unit {unit_id}")
                unit_data = test_data[test_data['unit_number'] == unit_id]
                true_rul = unit_data['RUL'].values
                features = unit_data.drop(columns=['unit_number', 'setting_1', 'setting_2', 'RUL']).values
                pred_rul = predict_rul(model, features, device)

                # Display results in a styled box
                st.markdown("""
                    <div class="result-box">
                    Mean Predicted RUL Fraction: {:.4f}<br>
                    Mean True RUL Fraction: {:.4f}
                    </div>
                """.format(np.mean(pred_rul), np.mean(true_rul)), unsafe_allow_html=True)

                # Download predictions
                results_df = pd.DataFrame({
                    'Cycle': unit_data['time_in_cycles'],
                    'True RUL': 1 - true_rul,
                    'Predicted RUL': 1 - pred_rul
                })
                st.download_button(
                    label="Download Predictions as CSV",
                    data=results_df.to_csv(index=False),
                    file_name=f"rul_predictions_unit_{unit_id}.csv",
                    mime="text/csv"
                )

            with col2:
                st.subheader("RUL Visualization")
                # Create Plotly line chart
                df = pd.DataFrame({
                    'Cycle': unit_data['time_in_cycles'],
                    'True RUL': 1 - true_rul,
                    'Predicted RUL': 1 - pred_rul
                })
                fig = px.line(
                    df, 
                    x='Cycle', 
                    y=['True RUL', 'Predicted RUL'], 
                    title=f'Predicted vs True RUL for Unit {unit_id}',
                    labels={'Cycle': 'Time in Cycles', 'value': 'Remaining Useful Life (RUL)', 'variable': 'Legend'},
                    color_discrete_map={
                        'True RUL': '#2ecc71',  # Green for true RUL
                        'Predicted RUL': '#e74c3c'  # Red for predicted RUL
                    }
                )
                fig.update_traces(
                    line=dict(width=2),
                    mode='lines+markers',
                    marker=dict(size=8)
                )
                fig.update_layout(
                    title_font_size=14,
                    xaxis_title_font_size=12,
                    yaxis_title_font_size=12,
                    legend_title_text='',
                    legend_font_size=10,
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Clean up
            os.remove(test_path)
            os.remove(rul_path)
            st.success("Analysis complete!")

if __name__ == "__main__":
    main()