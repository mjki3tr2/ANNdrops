from read_data import read_data
from read_fits import read_fits
from prep_data import prep_data
from calc_d32 import calc_d32
from remove_zeros import remove_zeros
from scale import scale
from split_data import split_data
from optimise_model import optimise_model
from plot_outputs import plot_outputs
from plot_training_history import plot_training_history
from inverse_scale import inverse_scale
from predict_and_inverse import predict_and_inverse
from plot_DSD_comp import plot_DSD_comp
from plot_xy import plot_xy

import numpy as np
from skopt.space import Real, Integer
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

import sys

"""
Set Parameters
"""
test_fraction = 0.1
validate_to_train_ratio = 0.3

#Model output options
#'linear' - no change
#'softmax' - make sure the sum is to 1
#'adjusted_softmax' - make sure the sum is to 1 with a threshold of 0.1 min
#'threshold_activation' - threshold of 0.1 min
#'normalized_relu' - make sure the sum is 1 by normalisation

n_runs = 5 # minimisation loops
num_optimise = 70 # number of minimisaion steps
num_initial = 20 # number of initial mapping guesses
space_model = [
    Real(1e-5, 1e-2,name='lr', prior='log-uniform'),
    Integer(500, 1000, name='epochs'),
    Integer(2, 3, name='layers'),
    Integer(10, 100, name='hidden_units'),
    Real(1.0, 2.0, name='taper_rate'),
    Real(0.0, 0.4, name='dropout_rate'),
    Real(1e-6, 1e-2, name='l2_factor', prior='log-uniform')
]

"""
Initial Setup
"""
# read in the data
DSD, diameter, properties, labels = read_data('Data_Summary2.csv')
fits, fit_labels = read_fits('Data_fits.csv')

# calculate the d32's of the data
d32_exp=np.zeros(len(DSD))
for index in range(len(DSD)):
    d32_exp[index] = calc_d32(DSD[index],diameter)

# Log the required values for the data, viscosity and Modes
properties_fix, _, _ = prep_data(properties,fits)

# Build the large new input properties array
diameter_fix = np.log(diameter.copy()) #diameters
properties_fix = np.asarray(properties_fix)  # shape [197, 8]
diameter_fix = np.asarray(diameter_fix).reshape(-1, 1)  # shape [101, 1]
props_repeated = np.repeat(properties_fix, repeats=len(diameter), axis=0)
diameter_tiled = np.tile(diameter_fix, (len(properties), 1))
combined = np.hstack((props_repeated, diameter_tiled))

# Build the new indices list
indices_DSD = combined[:,0].copy() # copy all ID values
indices_DSD = indices_DSD.reshape(-1, 1)

# Build the large output array
DSD = np.asarray(DSD)
DSD_flat = DSD.reshape(-1, 1)
DSD_flat = DSD_flat/100

# Scale the arrays
scaler_X_DSD, X_DSD_scaled = scale(combined[:,1:9])
scaler_y_DSD, y_DSD_scaled = scale(DSD_flat)

del DSD_flat, diameter_fix, properties_fix, props_repeated, diameter_tiled, combined

"""
Undertake the model fitting
"""
# split the data into test, train, and validate
XDSD_test, XDSD_train, XDSD_valid, yDSD_test, yDSD_train, yDSD_valid, idx_DSD_test, idx_DSD_train, idx_DSD_valid = split_data(X_DSD_scaled,y_DSD_scaled,indices_DSD,test_fraction,validate_to_train_ratio)

# undertake optimisation of the model for the volume fraction
print("Running DSD Optimisation")
final_model_DSD, history_DSD_final, combined_DSD_history, DSD_info = optimise_model(XDSD_train,yDSD_train,XDSD_valid,yDSD_valid,XDSD_test,yDSD_test,space_model,'relu',n_runs,num_optimise,num_initial)
final_model_DSD.save("DSD_model.keras")

# Get predictions for each split
yDSD_pred_train = final_model_DSD.predict(XDSD_train)
yDSD_pred_valid = final_model_DSD.predict(XDSD_valid)
yDSD_pred_test = final_model_DSD.predict(XDSD_test)
yDSD_pred_train, yDSD_pred_valid, yDSD_pred_test, yDSD_train, yDSD_valid, yDSD_test = inverse_scale(yDSD_pred_train,yDSD_pred_valid,yDSD_pred_test,yDSD_train,yDSD_valid,yDSD_test,scaler_y_DSD,log=False)

# plot the fraction output values
plot_outputs(yDSD_train[:, 0],yDSD_valid[:, 0],yDSD_test[:, 0],yDSD_pred_train[:, 0],yDSD_pred_valid[:, 0],yDSD_pred_test[:, 0],idx_DSD_train,idx_DSD_valid,idx_DSD_test,r'Plot for DSD',log=False, maerun=True,save_path='plots/DSDfull.png')

# plot the training history graph
plot_training_history(history_DSD_final, title=r'DSD Model Training History',save_path='plots/DSD_model_training.png')
plot_training_history(combined_DSD_history, title=r'DSD Model Training History',save_path='plots/DSD_model_combined.png')

# Re-run on the whole input to get the DSDs
y_DSD_pred = predict_and_inverse(final_model_DSD, X_DSD_scaled, scaler_y_DSD)
y_DSD_pred = 100*y_DSD_pred

# Change shape of the distributions to match original DSD
DSD_predicted = y_DSD_pred.reshape(len(properties), len(diameter))

# calculate the d32's of the data
d32_pred=np.zeros(len(DSD))
for index in range(len(DSD)):
    d32_pred[index] = calc_d32(DSD_predicted[index],diameter)

plot_xy(d32_exp,d32_pred,title_str=r'Plot for $d_{32}$',save_path='plots/d32DSD.png')

#plot DSDs
for i, index in enumerate(properties[:,0]):
    index=int(index.item())
    savepath=f'DSD_fits/{index:03d}.png'
    plot_DSD_comp(diameter,DSD,labels,properties,DSD_predicted,DSD_predicted,DSD_predicted,DSD_predicted,index,save_path=savepath)