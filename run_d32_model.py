from read_data import read_data
from read_fits import read_fits
from prep_data import prep_data
from scale import scale
from predict_and_inverse import predict_and_inverse
from calc_d32 import calc_d32
from optimise_parameters import optimise_parameters
from train_model import train_model
from plot_outputs import plot_outputs
from plot_training_history import plot_training_history

import numpy as np
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

import sys
import os

"""
Set Parameters
"""
test_fraction = 0.1

#Model output options
#'linear' - no change
#'softmax' - make sure the sum is to 1
#'adjusted_softmax' - make sure the sum is to 1 with a threshold of 0.1 min
#'threshold_activation' - threshold of 0.1 min
#'normalized_relu' - make sure the sum is 1 by normalisation

n_runs = 1 # minimisation loops
num_optimise = 70 # number of minimisaion steps
num_initial = 20 # number of initial mapping guesses
n_splits = 5 # number of K-fold sections
space_model = [
    Real(1e-5, 1e-2,name='lr', prior='log-uniform'),
    Integer(500, 1000, name='epochs'),
    Integer(1, 2, name='layers'),
    Integer(10, 50, name='hidden_units', prior='log-uniform'),
    Real(1.0, 2.0, name='taper_rate'),
    Real(0.0, 0.4, name='dropout_rate'),
    Real(1e-6, 1e-2, name='l2_factor', prior='log-uniform')
]
max_params_ratio = 100.0 # ratio of max number of parameters to number of samples

xd32_min=1
xd32_max=1000

train_min=1E-2
train_max=1

rng = np.random.RandomState() # Replace with 42 to get repeatable splits.

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
properties_fix, _, indices_d32 = prep_data(properties,fits)

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
# split the data into test, and the data for the K-fold
Xd32_data, Xd32_test, yd32_data, yd32_test, idx_d32_data, idx_d32_test = train_test_split(
    X_d32_scaled,y_d32_scaled,indices_d32,test_size=test_fraction,random_state=rng
)

# undertake optimisation of the model for the d32's
print("Running d32 Optimisation")
d32_info = optimise_parameters(Xd32_data,yd32_data,Xd32_test,yd32_test,space_model,'linear',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
final_model_d32, history_d32_final = train_model(Xd32_data,yd32_data,Xd32_test,yd32_test,'linear',d32_info)
final_model_d32.save("d32_model.keras")

# look at the model
final_model_d32.summary()
os.makedirs("plots", exist_ok=True)
plot_model(final_model_d32, to_file='plots/model_d32.png', show_shapes=True, show_layer_names=True)

# plot the training history graph
plot_training_history(history_d32_final, title=r'$d_{32}$ Model Training History', y_min=train_min, y_max=train_max, save_path='plots/d32_model_training.png')

# Get predictions for each split
yd32_pred_data = predict_and_inverse(final_model_d32, Xd32_data, scaler_y_d32, exp=True)
yd32_pred_test = predict_and_inverse(final_model_d32, Xd32_test, scaler_y_d32, exp=True)
yd32_data = np.exp(scaler_y_d32.inverse_transform(yd32_data))
yd32_test = np.exp(scaler_y_d32.inverse_transform(yd32_test))

# plot the d32 output values
plot_outputs(yd32_data,yd32_test,yd32_pred_data,yd32_pred_test,idx_d32_data,idx_d32_test,r'Plot for $d_{32}$',log=True, maerun=False, x_min=xd32_min, x_max=xd32_max, save_path='plots/d32model.png')