def predict_and_inverse(model, X, scaler=None, exp=False):
    import numpy as np
    y_pred = model.predict(X)
    if scaler:
        y_pred = scaler.inverse_transform(y_pred)
    return np.exp(y_pred) if exp else y_pred