import pandas as pd

def preprocess_data(df):
    df = df.dropna()
    X = pd.get_dummies(df.drop('Irrigation_Need', axis=1))
    y = df['Irrigation_Need']
    return X, y

def get_water_depth(prediction):
    """Converts AI label to mm of water depth."""
    mapping = {
        "Low": 2.5,    # 2.5 mm
        "Medium": 5.5, # 5.5 mm
        "High": 9.0    # 9.0 mm
    }
    return mapping.get(prediction, 0.0)