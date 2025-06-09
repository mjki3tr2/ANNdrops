from read_data import read_data
from read_fits import read_fits
from prep_data import prep_data
from remove_zeros import remove_zeros
from scale import scale
from optimise_parameters import optimise_parameters
from train_model import train_model
from plot_outputs import plot_outputs
from plot_training_history import plot_training_history
from predict_and_inverse import predict_and_inverse

import numpy as np
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

import sys
import os

"""
Information
This is the base file for runing the ANN fitting
The data is normalised and then a test set is selected
The ANN model parameters are then optimised using K-fold
This is undertaken for f, Mo1 to Mo3, and s1 to s3
"""

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

# models to run
run_f=True
run_Mo=False
run_s=False

"""
Initial Setup
"""
# read in the data
DSD, diameter, properties, labels = read_data('Data_Summary.csv')
fits, fit_labels = read_fits('Data_fits.csv')

# Log the required values for the data, viscosity and Modes
properties_fix, fits_fix, indices_f = prep_data(properties,fits)

#Sort the volume fraction parameters into inlet and outlet
X_f = properties_fix[:,1:8].copy()
y_f = fits_fix[:, [3, 6, 9]].copy()

# if f_i = 0, we do NOT want Mo_i or s_i in that row
X_1, y_Mo1, indices_1 = remove_zeros(properties_fix,fits_fix[:,3],fits_fix[:,1],fits_fix[:,0])
X_1, y_s1, indices_1 = remove_zeros(properties_fix,fits_fix[:,3],fits_fix[:,2],fits_fix[:,0])
X_2, y_Mo2, indices_2 = remove_zeros(properties_fix,fits_fix[:,6],fits_fix[:,4],fits_fix[:,0])
X_2, y_s2, indices_2 = remove_zeros(properties_fix,fits_fix[:,6],fits_fix[:,5],fits_fix[:,0])
X_3, y_Mo3, indices_3 = remove_zeros(properties_fix,fits_fix[:,9],fits_fix[:,7],fits_fix[:,0])
X_3, y_s3, indices_3 = remove_zeros(properties_fix,fits_fix[:,9],fits_fix[:,8],fits_fix[:,0])

# Scale X's,
scaler_X_f, X_f_scaled = scale(X_f)
scaler_X_1, X_1_scaled = scale(X_1)
scaler_X_2, X_2_scaled = scale(X_2)
scaler_X_3, X_3_scaled = scale(X_3)
del X_f, X_1,X_2, X_3 

# y_f does not need scaling as it is between 0 and 1 anyway
# Scale y_Mo's
scaler_y_Mo1, y_Mo1_scaled = scale(y_Mo1)
scaler_y_Mo2, y_Mo2_scaled = scale(y_Mo2)
scaler_y_Mo3, y_Mo3_scaled = scale(y_Mo3)
del y_Mo1, y_Mo2, y_Mo3

# Scale y_s's
scaler_y_s1, y_s1_scaled = scale(y_s1)
scaler_y_s2, y_s2_scaled = scale(y_s2)
scaler_y_s3, y_s3_scaled = scale(y_s3)
del y_s1, y_s2, y_s3

"""
Undertake the volume fraction fits
"""
if run_f:
    # split the data into test, and the data for the K-fold
    Xf_data, Xf_test, yf_data, yf_test, idx_f_data, idx_f_test = train_test_split(
        X_f_scaled,y_f,indices_f,test_size=test_fraction,random_state=123
    )
    
    # undertake optimisation of the model for the volume fraction
    print("Running Volume Fraction Optimisation")
    volfrac_info = optimise_parameters(Xf_data,yf_data,Xf_test,yf_test,space_model,'normalized_relu',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
    final_model_f, history_volfrac_final = train_model(Xf_data,yf_data,Xf_test,yf_test,'normalized_relu',volfrac_info)
    final_model_f.save("volume_fraction_model.keras")
    
    # look at the model
    final_model_f.summary()
    os.makedirs("plots", exist_ok=True)
    plot_model(final_model_f, to_file='plots/model_f.png', show_shapes=True, show_layer_names=True)
    
    # Get predictions for each split
    yf_pred_data = predict_and_inverse(final_model_f, Xf_data)
    yf_pred_test = predict_and_inverse(final_model_f, Xf_test)
    
    # plot the fraction output values
    plot_outputs(yf_data[:, 0],yf_test[:, 0],yf_pred_data[:, 0],yf_pred_test[:, 0],idx_f_data,idx_f_test,r'Plot for $f_1$',log=False, maerun=True,x_min=0,x_max=1,save_path='plots/f1.png')
    plot_outputs(yf_data[:, 1],yf_test[:, 1],yf_pred_data[:, 1],yf_pred_test[:, 1],idx_f_data,idx_f_test,r'Plot for $f_2$',log=False, maerun=True,x_min=0,x_max=1,save_path='plots/f2.png')
    plot_outputs(yf_data[:, 2],yf_test[:, 2],yf_pred_data[:, 2],yf_pred_test[:, 2],idx_f_data,idx_f_test,r'Plot for $f_3$',log=False, maerun=True,x_min=0,x_max=1,save_path='plots/f3.png')
    
    # plot the training history graph
    plot_training_history(history_volfrac_final, title=r'$f$ Model Training History',save_path='plots/f_model_training.png')

"""
Undertake the Mo fits
"""
if run_Mo:
    print("Running Mo1 Optimisation")
    # undertake optimisation of Mo1 model
    XMo1_data, XMo1_test, yMo1_data, yMo1_test, idx_Mo1_data, idx_Mo1_test = train_test_split(
        X_1_scaled,y_Mo1_scaled,indices_1,test_size=test_fraction,random_state=123
    )
    
    Mo1_info = optimise_parameters(XMo1_data,yMo1_data,XMo1_test,yMo1_test,space_model,'linear',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
    final_model_Mo1, history_Mo1_final = train_model(XMo1_data,yMo1_data,XMo1_test,yMo1_test,'linear',Mo1_info)
    final_model_Mo1.save("Mo1_model.keras")
    
    # look at the model
    final_model_Mo1.summary()
    os.makedirs("plots", exist_ok=True)
    plot_model(final_model_Mo1, to_file='plots/model_Mo1.png', show_shapes=True, show_layer_names=True)
    
    # Get predictions for each split
    yMo1_pred_data = predict_and_inverse(final_model_Mo1, XMo1_data, scaler_y_Mo1, exp=True)
    yMo1_pred_test = predict_and_inverse(final_model_Mo1, XMo1_test, scaler_y_Mo1, exp=True)
    yMo1_data = np.exp(scaler_y_Mo1.inverse_transform(yMo1_data))
    yMo1_test = np.exp(scaler_y_Mo1.inverse_transform(yMo1_test))
    
    # plot the Mo output values
    plot_outputs(yMo1_data,yMo1_test,yMo1_pred_data,yMo1_pred_test,idx_Mo1_data,idx_Mo1_test,r'Plot for Mo$_1$',log=True, maerun=False,x_min=1,x_max=1000,save_path='plots/Mo1.png')
    
    # plot the training history graph
    plot_training_history(history_Mo1_final, title=r'Mo$_1$ Model Training History',save_path='plots/Mo1_model_training.png')
    
    print("Running Mo2 Optimisation")
    XMo2_data, XMo2_test, yMo2_data, yMo2_test, idx_Mo2_data, idx_Mo2_test = train_test_split(
        X_2_scaled,y_Mo2_scaled,indices_2,test_size=test_fraction,random_state=123
    )
    
    Mo2_info = optimise_parameters(XMo2_data,yMo2_data,XMo2_test,yMo2_test,space_model,'linear',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
    final_model_Mo2, history_Mo2_final = train_model(XMo2_data,yMo2_data,XMo2_test,yMo2_test,'linear',Mo2_info)
    final_model_Mo2.save("Mo2_model.keras")
    
    # look at the model
    final_model_Mo2.summary()
    os.makedirs("plots", exist_ok=True)
    plot_model(final_model_Mo2, to_file='plots/model_Mo2.png', show_shapes=True, show_layer_names=True)
    
    # Get predictions for each split
    yMo2_pred_data = predict_and_inverse(final_model_Mo2, XMo2_data, scaler_y_Mo2, exp=True)
    yMo2_pred_test = predict_and_inverse(final_model_Mo2, XMo2_test, scaler_y_Mo2, exp=True)
    yMo2_data = np.exp(scaler_y_Mo2.inverse_transform(yMo2_data))
    yMo2_test = np.exp(scaler_y_Mo2.inverse_transform(yMo2_test))
    
    # plot the Mo output values
    plot_outputs(yMo2_data,yMo2_test,yMo2_pred_data,yMo2_pred_test,idx_Mo2_data,idx_Mo2_test,r'Plot for Mo$_2$',log=True, maerun=False,x_min=1,x_max=1000,save_path='plots/Mo2.png')
    
    # plot the training history graph
    plot_training_history(history_Mo2_final, title=r'Mo$_2$ Model Training History',save_path='plots/Mo2_model_training.png')
        
    print("Running Mo3 Optimisation")
    XMo3_data, XMo3_test, yMo3_data, yMo3_test, idx_Mo3_data, idx_Mo3_test = train_test_split(
        X_3_scaled,y_Mo3_scaled,indices_3,test_size=test_fraction,random_state=123
    )
    
    Mo3_info = optimise_parameters(XMo3_data,yMo3_data,XMo3_test,yMo3_test,space_model,'linear',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
    final_model_Mo3, history_Mo3_final = train_model(XMo3_data,yMo3_data,XMo3_test,yMo3_test,'linear',Mo3_info)
    final_model_Mo3.save("Mo1_model.keras")
    
    # look at the model
    final_model_Mo3.summary()
    os.makedirs("plots", exist_ok=True)
    plot_model(final_model_Mo3, to_file='plots/model_Mo3.png', show_shapes=True, show_layer_names=True)
    
    # Get predictions for each split
    yMo3_pred_data = predict_and_inverse(final_model_Mo3, XMo3_data, scaler_y_Mo3, exp=True)
    yMo3_pred_test = predict_and_inverse(final_model_Mo3, XMo3_test, scaler_y_Mo3, exp=True)
    yMo3_data = np.exp(scaler_y_Mo3.inverse_transform(yMo3_data))
    yMo3_test = np.exp(scaler_y_Mo3.inverse_transform(yMo3_test))
    
    # plot the Mo output values
    plot_outputs(yMo3_data,yMo3_test,yMo3_pred_data,yMo3_pred_test,idx_Mo3_data,idx_Mo3_test,r'Plot for Mo$_3$',log=True, maerun=False,x_min=1,x_max=1000,save_path='plots/Mo3.png')
    
    # plot the training history graph
    plot_training_history(history_Mo3_final, title=r'Mo$_3$ Model Training History',save_path='plots/Mo3_model_training.png')
    
"""
Undertake the s fits
"""
if run_s:
    # undertake optimisation of the s1 model
    print("Running s1 Optimisation")
    Xs1_data, Xs1_test, ys1_data, ys1_test, idx_s1_data, idx_s1_test = train_test_split(
        X_1_scaled,y_s1_scaled,indices_1,test_size=test_fraction,random_state=123
    )
    
    s1_info = optimise_parameters(Xs1_data,ys1_data,Xs1_test,ys1_test,space_model,'linear',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
    final_model_s1, history_s1_final = train_model(Xs1_data,ys1_data,Xs1_test,ys1_test,'linear',s1_info)
    final_model_s1.save("s1_model.keras")
    
    # look at the model
    final_model_s1.summary()
    os.makedirs("plots", exist_ok=True)
    plot_model(final_model_s1, to_file='plots/model_s1.png', show_shapes=True, show_layer_names=True)
    
    # Get predictions for each split
    ys1_pred_data = predict_and_inverse(final_model_s1, Xs1_data, scaler_y_s1)
    ys1_pred_test = predict_and_inverse(final_model_s1, Xs1_test, scaler_y_s1)
    ys1_data = scaler_y_s1.inverse_transform(ys1_data)
    ys1_test = scaler_y_s1.inverse_transform(ys1_test)
    
    # plot the s output values
    plot_outputs(ys1_data,ys1_test,ys1_pred_data,ys1_pred_test,idx_s1_data,idx_s1_test,r'Plot for $\sigma_1$',log=False, maerun=False,x_min=0.2,x_max=1.3,save_path='plots/s1.png')
    
    # plot the training history graph
    plot_training_history(history_s1_final, title=r'$\sigma_1$ Model Training History',save_path='plots/s1_model_training.png')
    
    # undertake optimisation of the s2 model
    print("Running s2 Optimisation")
    Xs2_data, Xs2_test, ys2_data, ys2_test, idx_s2_data, idx_s2_test = train_test_split(
        X_2_scaled,y_s2_scaled,indices_2,test_size=test_fraction,random_state=123
    )
    
    s2_info = optimise_parameters(Xs2_data,ys2_data,Xs2_test,ys2_test,space_model,'linear',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
    final_model_s2, history_s2_final = train_model(Xs2_data,ys2_data,Xs2_test,ys2_test,'linear',s2_info)
    final_model_s2.save("s2_model.keras")
    
    # look at the model
    final_model_s2.summary()
    os.makedirs("plots", exist_ok=True)
    plot_model(final_model_s2, to_file='plots/model_s2.png', show_shapes=True, show_layer_names=True)
    
    # Get predictions for each split
    ys2_pred_data = predict_and_inverse(final_model_s2, Xs2_data, scaler_y_s2)
    ys2_pred_test = predict_and_inverse(final_model_s2, Xs2_test, scaler_y_s2)
    ys2_data = scaler_y_s2.inverse_transform(ys2_data)
    ys2_test = scaler_y_s2.inverse_transform(ys2_test)
    
    # plot the s output values
    plot_outputs(ys2_data,ys2_test,ys2_pred_data,ys2_pred_test,idx_s2_data,idx_s2_test,r'Plot for $\sigma_2$',log=False, maerun=False,x_min=0.2,x_max=1.3,save_path='plots/s2.png')
    
    # plot the training history graph
    plot_training_history(history_s2_final, title=r'$\sigma_2$ Model Training History',save_path='plots/s2_model_training.png')
    
    # undertake optimisation of the s3 model
    print("Running s3 Optimisation")
    Xs3_data, Xs3_test, ys3_data, ys3_test, idx_s3_data, idx_s3_test = train_test_split(
        X_3_scaled,y_s3_scaled,indices_3,test_size=test_fraction,random_state=123
    )
    
    s3_info = optimise_parameters(Xs3_data,ys3_data,Xs3_test,ys3_test,space_model,'linear',n_runs,num_optimise,num_initial,n_splits,max_params_ratio)
    final_model_s3, history_s3_final = train_model(Xs3_data,ys3_data,Xs3_test,ys3_test,'linear',s3_info)
    final_model_s3.save("s3_model.keras")
    
    # look at the model
    final_model_s3.summary()
    os.makedirs("plots", exist_ok=True)
    plot_model(final_model_s3, to_file='plots/model_s3.png', show_shapes=True, show_layer_names=True)
    
    # Get predictions for each split
    ys3_pred_data = predict_and_inverse(final_model_s3, Xs3_data, scaler_y_s3)
    ys3_pred_test = predict_and_inverse(final_model_s3, Xs3_test, scaler_y_s3)
    ys3_data = scaler_y_s3.inverse_transform(ys3_data)
    ys3_test = scaler_y_s3.inverse_transform(ys3_test)
    
    # plot the s output values
    plot_outputs(ys3_data,ys3_test,ys3_pred_data,ys3_pred_test,idx_s3_data,idx_s3_test,r'Plot for $\sigma_3$',log=False, maerun=False,x_min=0.2,x_max=1.3,save_path='plots/s3.png')
    
    # plot the training history graph
    plot_training_history(history_s3_final, title=r'$\sigma_3$ Model Training History',save_path='plots/s3_model_training.png')
