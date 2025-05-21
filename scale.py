def scale(X):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    return scaler_X, X_scaled