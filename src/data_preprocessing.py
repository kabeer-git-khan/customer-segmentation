import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_data(path):
    ext = os.path.splitext(path)[-1]
    if ext == '.csv':
        return pd.read_csv(path, encoding='ISO-8859-1')
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format")

def preprocess_data(df):
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    RFM = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    RFM.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    RFM_scaled = scaler.fit_transform(RFM[['Recency', 'Frequency', 'Monetary']])
    return RFM, RFM_scaled
