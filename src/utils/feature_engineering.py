import numpy as np

def engineer_features(N, P, K, temperature, humidity, ph, rainfall):
    """
    Given the base 7 features, computes additional engineered features.
    Returns a numpy array of all features.
    Base features: N, P, K, temperature, humidity, ph, rainfall
    Engineered features:
    - N/P ratio
    - N/K ratio
    - P/K ratio
    - Temp * Humidity (Heat Index approximation)
    """
    epsilon = 1e-5
    np_ratio = N / (P + epsilon)
    nk_ratio = N / (K + epsilon)
    pk_ratio = P / (K + epsilon)
    temp_humidity = temperature * humidity
    
    return np.array([
        N, P, K, temperature, humidity, ph, rainfall,
        np_ratio, nk_ratio, pk_ratio, temp_humidity
    ])

def engineer_features_df(df):
    """
    Applies feature engineering to a pandas DataFrame.
    """
    df_new = df.copy()
    epsilon = 1e-5
    df_new['np_ratio'] = df_new['N'] / (df_new['P'] + epsilon)
    df_new['nk_ratio'] = df_new['N'] / (df_new['K'] + epsilon)
    df_new['pk_ratio'] = df_new['P'] / (df_new['K'] + epsilon)
    df_new['temp_humidity'] = df_new['temperature'] * df_new['humidity']
    return df_new
