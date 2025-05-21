def split_data(x,y,indices,split1,split2):
    
    from sklearn.model_selection import train_test_split
    
    # Will perform the splits while keeping track of indices
    # First, split off the test set 
    x_temp, x_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        x, y, indices, test_size=split1,random_state=123
    )
    # Then, split the remainder into training and validation
    x_train, x_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
        x_temp, y_temp, idx_temp, test_size=split2, random_state=123
    )
    # Now we have 3 arrays (idx_f_train, idx_f_valid, idx_f_test) that tell us which rows from the original df belong to each set
    return x_test, x_train, x_valid, y_test, y_train, y_valid, idx_test, idx_train, idx_valid