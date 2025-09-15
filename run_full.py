from read_data import read_data
from read_fits import read_fits
from prep_data import prep_data
from calc_d32 import calc_d32
from scale import scale
from optimise_parameters import optimise_parameters
from train_model import train_model
from predict_and_inverse import predict_and_inverse
from plot_outputs import plot_outputs
from plot_training_history import plot_training_history
from plot_DSD_comp import plot_DSD_comp
from plot_xy import plot_xy

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
    Integer(1, 3, name='layers'),
    Integer(10, 250, name='hidden_units'),
    Real(1.0, 2.0, name='taper_rate'),
    Real(0.0, 0.4, name='dropout_rate'),
    Real(1e-6, 1e-2, name='l2_factor', prior='log-uniform')
]
max_params_ratio = 100.0 # ratio of max number of parameters to number of samples

xDSD_min=0
xDSD_max=0.2
xd32_min=1
xd32_max=1000

train_min=1E-2
train_max=1

rng = np.random.RandomState() # Replace with 42 to get repeatable splits.

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
# split the data into test, and the data for the K-fold
XDSD_data, XDSD_test, yDSD_data, yDSD_test, idx_DSD_data, idx_DSD_test = train_test_split(
    X_DSD_scaled,y_DSD_scaled,indices_DSD,test_size=test_fraction,random_state=rng)

# undertake optimisation of the model for the volume fraction
print("Running DSD Optimisation")
DSD_info = optimise_parameters(XDSD_data,yDSD_data,XDSD_test,yDSD_test,space_model,'relu',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
final_model_DSD, history_DSD_final = train_model(XDSD_data,yDSD_data,XDSD_test,yDSD_test,'relu',DSD_info)
final_model_DSD.save("DSD_model.keras")

# look at the model
final_model_DSD.summary()
os.makedirs("plots", exist_ok=True)
plot_model(final_model_DSD, to_file='plots/model_DSD.png', show_shapes=True, show_layer_names=True)

# plot the training history graph
plot_training_history(history_DSD_final, title=r'DSD Model Training History', y_min=train_min, y_max=train_max, save_path='plots/DSD_model_training.png')

# Get predictions for each split
yDSD_pred_data = predict_and_inverse(final_model_DSD, XDSD_data, scaler_y_DSD)
yDSD_pred_test = predict_and_inverse(final_model_DSD, XDSD_test, scaler_y_DSD)
yDSD_data = scaler_y_DSD.inverse_transform(yDSD_data)
yDSD_test = scaler_y_DSD.inverse_transform(yDSD_test)

# plot the fraction output values
plot_outputs(yDSD_data,yDSD_test,yDSD_pred_data,yDSD_pred_test,idx_DSD_data,idx_DSD_test,r'Plot for DSD points',log=False, maerun=True, x_min=xDSD_min, x_max=xDSD_max, save_path='plots/DSDmodel.png')

# Re-run on the whole input to get the DSDs
y_DSD_pred = predict_and_inverse(final_model_DSD, X_DSD_scaled, scaler_y_DSD)
y_DSD_pred = 100*y_DSD_pred

# Change shape of the distributions to match original DSD
DSD_predicted = y_DSD_pred.reshape(len(properties), len(diameter))

# calculate the d32's of the data
d32_pred=np.zeros(len(DSD))
for index in range(len(DSD)):
    d32_pred[index] = calc_d32(DSD_predicted[index],diameter)

plot_xy(d32_exp,d32_pred,title_str=r'Plot for $d_{32}$',index=properties[:,0],x_min=xd32_min, x_max=xd32_max, save_path='plots/d32DSD.png')

#plot DSDs
for i, index in enumerate(properties[:,0]):
    index=int(index.item())
    savepath=f'DSD_fits/{index:03d}.png'
    plot_DSD_comp(diameter,DSD,labels,properties,DSD_predicted,DSD_predicted,DSD_predicted,DSD_predicted,i,save_path=savepath)
