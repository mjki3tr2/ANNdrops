def inverse_scale(y_pred_train,y_pred_valid,y_pred_test,y_train,y_valid,y_test,scaler_y,
                  log=True):
    
    import numpy as np
    
    if log:
        y_pred_train = np.exp(scaler_y.inverse_transform(y_pred_train))
        y_pred_valid = np.exp(scaler_y.inverse_transform(y_pred_valid))
        y_pred_test = np.exp(scaler_y.inverse_transform(y_pred_test))
        y_train = np.exp(scaler_y.inverse_transform(y_train))
        y_valid = np.exp(scaler_y.inverse_transform(y_valid))
        y_test = np.exp(scaler_y.inverse_transform(y_test))
    else:
        y_pred_train = scaler_y.inverse_transform(y_pred_train)
        y_pred_valid = scaler_y.inverse_transform(y_pred_valid)
        y_pred_test = scaler_y.inverse_transform(y_pred_test)
        y_train = scaler_y.inverse_transform(y_train)
        y_valid = scaler_y.inverse_transform(y_valid)
        y_test = scaler_y.inverse_transform(y_test)
    
    return y_pred_train, y_pred_valid, y_pred_test, y_train, y_valid, y_test 