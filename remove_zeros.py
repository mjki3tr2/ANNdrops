def remove_zeros(properties,zero_in,vals_in,ind_in):
    import numpy as np
    
    y=[]
    indices=[]
    X=[]
    for i in range(len(properties)):
        if zero_in[i] != 0: #f1
            y.append(vals_in[i]) #Mo1
            indices.append(ind_in[i]) #ID values
            X.append(properties[i,1:8]) # copy all properties
    
    y = np.array(y).reshape(-1, 1)
    indices = np.array(indices).reshape(-1, 1)
    X = np.array(X)
    
    return X, y, indices