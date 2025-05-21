def read_data(csv_file_path):
    
    import sys
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as sp
    import mglearn
    import sklearn
    import csv
    from scipy.interpolate import make_interp_spline, BSpline
    
    #csv_file_path = 'Data_Summary.csv'
    
    # Initialize an empty list to hold the data for each column
    properties = []
    DSD = []
    
    # Open the CSV file using a context manager
    with open(csv_file_path, 'r', newline='') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file) 
    
        # Iterate through the remaining rows in the CSV file
        for row_number, row in enumerate(csv_reader):
                
            if row_number < 8:
                # Check if the number of columns in the current row matches the number of columns found so far
                if len(properties) == 0:    
                    # If no columns have been found yet, create an empty list for each column
                    properties = [[] for _ in range(len(row))]
                
                # Append each value to the respective column's list
                for j, value in enumerate(row):
                    if j == 0:
                        properties[j].append(value)
                    else:
                        if row_number == 0:
                            properties[j].append(int(value))
                        else:
                            properties[j].append(float(value))
            else:
                # Check if the number of columns in the current row matches the number of columns found so far
                if len(DSD) == 0:
                    # If no columns have been found yet, create an empty list for each column
                    DSD = [[] for _ in range(len(row))]
                    
                # Append each value to the respective column's list
                for i, value in enumerate(row):
                    DSD[i].append(float(value))
    
    diameter = DSD[0]
    diameter = np.array(diameter)
    diameter = diameter.reshape(-1, 1)
    del DSD[0]
    DSD = np.array(DSD)
    labels = properties[0]
    del properties[0]
    properties = np.array(properties)
    
    return DSD, diameter, properties, labels