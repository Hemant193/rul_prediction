import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import os
from utils import RULModel, preprocess_test_data, predict_rul, COLUMNS

def main():
    st.title("Aircraft Engine RUL Prediction")
    st.write("Upload CMAPSS test data and RUL files to predict Remaining Useful Life.")

    # Uploading the files
    test_file = st.file_uploader("Upload Test Data (e.g., test_FD001.txt)", type=["txt"])
    rul_file = st.file_uploader("Upload RUL Data (e.g., RUL_FD001.txt)", type=["txt"])

    if test_file and rul_file:
        # Save uploaded files temporarily
        test_path = "temp_test.txt"
        rul_path = "temp_rul.txt"
        with open(test_path, "wb") as f:
            f.write(test_file.read())
        with open(rul_path, "wb") as f:
            f.write(rul_file.read())

        # Loading the scaler and model
        try:
            with open("models/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            model = RULModel(input_dim=16)
            model.load_state_dict(torch.load("models/rul_ann_model.pth", map_location="cpu"))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
        except FileNotFoundError:
            st.error("Model or scaler not found. Run the notebook to train and save them.")
            return
        except Exception as e:
            st.error(f"Error loading model/scaler: {e}")
            return

        # Preprocessing the data
        try:
            test_data = preprocess_test_data(test_path, rul_path, scaler)
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return

        # Select unit number
        unit_ids = test_data['unit_number'].unique()
        unit_id = st.selectbox("Select Unit Number", unit_ids)

        # Filter and predict
        unit_data = test_data[test_data['unit_number'] == unit_id]
        true_rul = unit_data['RUL'].values
        features = unit_data.drop(columns=['unit_number', 'setting_1', 'setting_2', 'RUL']).values
        pred_rul = predict_rul(model, features, device)

        # Display results
        st.subheader(f"RUL Predictions for Unit {unit_id}")
        st.write(f"Mean Predicted RUL Fraction: {np.mean(pred_rul):.4f}")
        st.write(f"Mean True RUL Fraction: {np.mean(true_rul):.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(unit_data['time_in_cycles'], 1 - true_rul, label='True RUL', marker='o')
        ax.plot(unit_data['time_in_cycles'], 1 - pred_rul, label='Predicted RUL', marker='s')
        ax.set_xlabel('Time in Cycles')
        ax.set_ylabel('Remaining Useful Life (RUL)')
        ax.set_title(f'Predicted vs True RUL for Unit {unit_id}')
        ax.legend()
        st.pyplot(fig)

        # Clean up
        os.remove(test_path)
        os.remove(rul_path)

if __name__ == "__main__":
    main()