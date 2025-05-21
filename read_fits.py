def read_fits(csv_file_path):
    
    import csv
    import numpy as np
        
    #csv_file_path = 'Data_fits.csv'
    
    # Initialize an empty list to hold the data for each column
    fits = []
    
    # Open the CSV file using a context manager
    with open(csv_file_path, 'r', newline='') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file) 
    
        # Iterate through the remaining rows in the CSV file
        for row_number, row in enumerate(csv_reader):
                
            # Check if the number of columns in the current row matches the number of columns found so far
            if len(fits) == 0:
                # If no columns have been found yet, create an empty list for each column
                fits = [[] for _ in range(len(row))]
                    
            # Append each value to the respective column's list
            for i, value in enumerate(row):
                if i == 0:
                    fits[i].append(value)
                else:
                    if row_number == 0:
                        fits[i].append(int(value))
                    else:
                        fits[i].append(float(value))
    
    fit_labels = fits[0]
    del fits[0]
    fits = np.array(fits)
    
    # Convert volume fractions from % to [0,1]
    fits[:,3] = fits[:,3] / 100.0 #f1
    fits[:,6] = fits[:,6] / 100.0 #f2
    fits[:,9] = fits[:,9] / 100.0 #f3
    
    return fits, fit_labels