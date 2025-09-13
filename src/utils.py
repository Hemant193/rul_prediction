import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Column names for CMAPSS dataset
COLUMNS = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'TRA', 'T2', 'T24', 'T30', 'T50', 'P2', 
           'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 
           'PCNfR_dmd', 'W31', 'W32']

class RULDataset(Dataset):
    def __init__(self, dataframe):
        self.X = dataframe.drop(columns=["RUL"]).values.astype(np.float32)
        self.y = dataframe["RUL"].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class RULModel(nn.Module):
    def __init__(self, input_dim):
        super(RULModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

def preprocess_test_data(test_path, rul_path, scaler):
    test_data_rul = pd.read_csv(rul_path, sep=" ", header=None)
    test_data = pd.read_csv(test_path, sep=" ", header=None)
    test_data.drop(columns=[26, 27], inplace=True)
    test_data.columns = COLUMNS
    test_data.drop(columns=['Nf_dmd', 'PCNfR_dmd', 'P2', 'T2', 'TRA', 'farB', 'epr'], inplace=True)
    
    eol = []
    for un in test_data['unit_number'].unique():
        temp_data = test_data[test_data['unit_number'] == un]
        eol_temp = test_data_rul.iloc[un-1].values[0]
        eol_temp_list = [eol_temp for _ in range(len(temp_data))]
        eol.extend(eol_temp_list)
    
    test_data['RUL'] = test_data['time_in_cycles'].values / (
        test_data.groupby('unit_number')['time_in_cycles'].transform('max').values + eol)
    
    features = test_data.drop(columns=['unit_number', 'setting_1', 'setting_2', 'RUL'])
    test_data.loc[:, features.columns] = scaler.transform(features)
    
    return test_data

def predict_rul(model, data, device):
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(data_tensor).cpu().numpy().flatten()
    return predictions