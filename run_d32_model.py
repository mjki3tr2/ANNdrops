from read_data import read_data
from read_fits import read_fits
from prep_data import prep_data
from remove_zeros import remove_zeros
from scale import scale
from predict_and_inverse import predict_and_inverse
from split_data import split_data
from calc_d32 import calc_d32
from optimise_model import optimise_model
from plot_outputs import plot_outputs
from plot_training_history import plot_training_history
from inverse_scale import inverse_scale

import numpy as np
from skopt.space import Real, Integer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model

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
    Integer(1, 2, name='layers'),
    Integer(10, 100, name='hidden_units'),
    Real(1.0, 2.0, name='taper_rate'),
    Real(0.0, 0.4, name='dropout_rate'),
    Real(1e-6, 1e-2, name='l2_factor', prior='log-uniform')
]

"""
Initial Setup
"""
# read in the data
DSD, diameter, properties, labels = read_data('Data_Summary.csv')
fits, fit_labels = read_fits('Data_fits.csv')

# calculate the d32's of the data
d32_exp=np.zeros(len(DSD))
for index in range(len(DSD)):
    d32_exp[index] = calc_d32(DSD[index],diameter)

# Log the required values for the data, viscosity and Modes
properties_fix, fits_fix, indices_d32 = prep_data(properties,fits)
del fits_fix

fits_fix = np.log(d32_exp).copy()

# Sort the d32 fraction parameters into inlet and outlet
X_d32 = properties_fix[:,1:8].copy() # copy all properties
y_d32 = fits_fix.copy() # f1
y_d32 = y_d32.reshape(-1, 1)

# Scale values
scaler_X_d32, X_d32_scaled = scale(X_d32)
scaler_y_d32, y_d32_scaled = scale(y_d32)

"""
Undertake the d32 fits
"""
# split the data into test, train, and validate
Xd32_test, Xd32_train, Xd32_valid, yd32_test, yd32_train, yd32_valid, idx_d32_test, idx_d32_train, idx_d32_valid = split_data(X_d32_scaled,y_d32_scaled,indices_d32,test_fraction,validate_to_train_ratio)

## undertake optimisation of the model for the volume fraction
print("Running d32 Optimisation")
final_model_d32, history_d32_final, combined_d32_history, d32_info = optimise_model(Xd32_train,yd32_train,Xd32_valid,yd32_valid,Xd32_test,yd32_test,space_model,'linear',n_runs,num_optimise,num_initial)
final_model_d32.save("d32_model.keras")

# plot the training history graph
plot_training_history(history_d32_final, title=r'$d_{32}$ Model Training History',save_path='plots/d32_model_training.png')
plot_training_history(combined_d32_history, title=r'$d_{32}$ Model Training History',save_path='plots/d32_model_combined.png')

# Get predictions for each split
yd32_pred_train = final_model_d32.predict(Xd32_train)
yd32_pred_valid = final_model_d32.predict(Xd32_valid)
yd32_pred_test = final_model_d32.predict(Xd32_test)

yd32_pred_train, yd32_pred_valid, yd32_pred_test, yd32_train, yd32_valid, yd32_test = inverse_scale(yd32_pred_train,yd32_pred_valid,yd32_pred_test,yd32_train,yd32_valid,yd32_test,scaler_y_d32)

# plot the fraction output values
plot_outputs(yd32_train,yd32_valid,yd32_test,yd32_pred_train,yd32_pred_valid,yd32_pred_test,idx_d32_train,idx_d32_valid,idx_d32_test,r'Plot for $d_{32}$',log=True, maerun=False, save_path='plots/d32model.png')

final_model_d32.summary()
plot_model(final_model_d32, to_file='plots/model_d32.png', show_shapes=True, show_layer_names=True,rankdir='TB',dpi=600)