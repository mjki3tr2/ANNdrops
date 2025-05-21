from read_data import read_data
from read_fits import read_fits
from prep_data import prep_data
from remove_zeros import remove_zeros
from scale import scale
from split_data import split_data
from optimise_model import optimise_model
from plot_outputs import plot_outputs
from plot_training_history import plot_training_history
from inverse_scale import inverse_scale

import numpy as np
from skopt.space import Real, Integer
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
    Integer(2, 3, name='layers'),
    Integer(10, 100, name='hidden_units'),
    Real(1.0, 2.0, name='taper_rate'),
    Real(0.0, 0.4, name='dropout_rate'),
    Real(1e-6, 1e-2, name='l2_factor', prior='log-uniform')
]

# models to run
run_f=True
run_Mo=True
run_s=True

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
    # split the data into test, train, and validate
    Xf_test, Xf_train, Xf_valid, yf_test, yf_train, yf_valid, idx_f_test, idx_f_train, idx_f_valid = split_data(X_f_scaled,y_f,indices_f,test_fraction,validate_to_train_ratio)
    
    # undertake optimisation of the model for the volume fraction
    print("Running Volume Fraction Optimisation")
    final_model_f, history_volfrac_final, combined_volfrac_history, volfrac_info = optimise_model(Xf_train,yf_train,Xf_valid,yf_valid,Xf_test,yf_test,space_model,'normalized_relu',n_runs,num_optimise,num_initial)
    final_model_f.save("volume_fraction_model.keras")
    
    # look at the model
    #final_model_f.summary()
    #plot_model(final_model_f, to_file='plots/model_f.png', show_shapes=True, show_layer_names=True)
    
    # Get predictions for each split
    yf_pred_train = final_model_f.predict(Xf_train)
    yf_pred_valid = final_model_f.predict(Xf_valid)
    yf_pred_test = final_model_f.predict(Xf_test)
    
    # plot the fraction output values
    plot_outputs(yf_train[:, 0],yf_valid[:, 0],yf_test[:, 0],yf_pred_train[:, 0],yf_pred_valid[:, 0],yf_pred_test[:, 0],idx_f_train,idx_f_valid,idx_f_test,r'Plot for $f_1$',log=False, maerun=True,save_path='plots/f1.png')
    plot_outputs(yf_train[:, 1],yf_valid[:, 1],yf_test[:, 1],yf_pred_train[:, 1],yf_pred_valid[:, 1],yf_pred_test[:, 1],idx_f_train,idx_f_valid,idx_f_test,r'Plot for $f_2$',log=False, maerun=True,save_path='plots/f2.png')
    plot_outputs(yf_train[:, 2],yf_valid[:, 2],yf_test[:, 2],yf_pred_train[:, 2],yf_pred_valid[:, 2],yf_pred_test[:, 2],idx_f_train,idx_f_valid,idx_f_test,r'Plot for $f_3$',log=False, maerun=True,save_path='plots/f3.png')
    
    # plot the training history graph
    plot_training_history(history_volfrac_final, title=r'$f$ Model Training History',save_path='plots/f_model_training.png')
    plot_training_history(combined_volfrac_history, title=r'$f$ Model Training History',save_path='plots/f_model_combined.png')

"""
Undertake the Mo fits
"""
if run_Mo:
    # undertake optimisation of Mo1 model
    XMo1_test, XMo1_train, XMo1_valid, yMo1_test, yMo1_train, yMo1_valid, idx_Mo1_test, idx_Mo1_train, idx_Mo1_valid = split_data(X_1_scaled,y_Mo1_scaled,indices_1,test_fraction,validate_to_train_ratio)
    print("Running Mo1 Optimisation")
    final_model_Mo1, history_Mo1_final, combined_Mo1_history, Mo1_info = optimise_model(XMo1_train,yMo1_train,XMo1_valid,yMo1_valid,XMo1_test,yMo1_test,space_model,'linear',n_runs,num_optimise,num_initial)
    final_model_Mo1.save("Mo1_model.keras")
    
    # Get predictions for each split
    yMo1_pred_train = final_model_Mo1.predict(XMo1_train)
    yMo1_pred_valid = final_model_Mo1.predict(XMo1_valid)
    yMo1_pred_test = final_model_Mo1.predict(XMo1_test)
    yMo1_pred_train, yMo1_pred_valid, yMo1_pred_test, yMo1_train, yMo1_valid, yMo1_test = inverse_scale(yMo1_pred_train,yMo1_pred_valid,yMo1_pred_test,yMo1_train,yMo1_valid,yMo1_test,scaler_y_Mo1)
    
    # plot the Mo output values
    plot_outputs(yMo1_train,yMo1_valid,yMo1_test,yMo1_pred_train,yMo1_pred_valid,yMo1_pred_test,idx_Mo1_train,idx_Mo1_valid,idx_Mo1_test,r'Plot for Mo$_1$',log=True, maerun=False,save_path='plots/Mo1.png')
    
    # plot the training history graph
    plot_training_history(history_Mo1_final, title=r'Mo$_1$ Model Training History',save_path='plots/Mo1_model_training.png')
    plot_training_history(combined_Mo1_history, title=r'Mo$_1$ Model Training History',save_path='plots/Mo1_model_combined.png')
    
    # undertake optimisation of Mo2 model
    XMo2_test, XMo2_train, XMo2_valid, yMo2_test, yMo2_train, yMo2_valid, idx_Mo2_test, idx_Mo2_train, idx_Mo2_valid = split_data(X_2_scaled,y_Mo2_scaled,indices_2,test_fraction,validate_to_train_ratio)
    print("Running Mo2 Optimisation")
    final_model_Mo2, history_Mo2_final, combined_Mo2_history, Mo2_info = optimise_model(XMo2_train,yMo2_train,XMo2_valid,yMo2_valid,XMo2_test,yMo2_test,space_model,'linear',n_runs,num_optimise,num_initial)
    final_model_Mo2.save("Mo2_model.keras")
    
    # Get predictions for each split
    yMo2_pred_train = final_model_Mo2.predict(XMo2_train)
    yMo2_pred_valid = final_model_Mo2.predict(XMo2_valid)
    yMo2_pred_test = final_model_Mo2.predict(XMo2_test)
    yMo2_pred_train, yMo2_pred_valid, yMo2_pred_test, yMo2_train, yMo2_valid, yMo2_test = inverse_scale(yMo2_pred_train,yMo2_pred_valid,yMo2_pred_test,yMo2_train,yMo2_valid,yMo2_test,scaler_y_Mo2)
    
    # plot the Mo output values
    plot_outputs(yMo2_train,yMo2_valid,yMo2_test,yMo2_pred_train,yMo2_pred_valid,yMo2_pred_test,idx_Mo2_train,idx_Mo2_valid,idx_Mo2_test,r'Plot for Mo$_2$',log=True, maerun=False,save_path='plots/Mo2.png')
    
    # plot the training history graph
    plot_training_history(history_Mo2_final, title=r'Mo$_2$ Model Training History',save_path='plots/Mo2_model_training.png')
    plot_training_history(combined_Mo2_history, title=r'Mo$_2$ Model Training History',save_path='plots/Mo2_model_combined.png')
    
    # undertake optimisation of Mo3 model
    XMo3_test, XMo3_train, XMo3_valid, yMo3_test, yMo3_train, yMo3_valid, idx_Mo3_test, idx_Mo3_train, idx_Mo3_valid = split_data(X_3_scaled,y_Mo3_scaled,indices_3,test_fraction,validate_to_train_ratio)
    print("Running Mo3 Optimisation")
    final_model_Mo3, history_Mo3_final, combined_Mo3_history, Mo3_info = optimise_model(XMo3_train,yMo3_train,XMo3_valid,yMo3_valid,XMo3_test,yMo3_test,space_model,'linear',n_runs,num_optimise,num_initial)
    final_model_Mo3.save("Mo3_model.keras")
    
    # Get predictions for each split
    yMo3_pred_train = final_model_Mo3.predict(XMo3_train)
    yMo3_pred_valid = final_model_Mo3.predict(XMo3_valid)
    yMo3_pred_test = final_model_Mo3.predict(XMo3_test)
    yMo3_pred_train, yMo3_pred_valid, yMo3_pred_test, yMo3_train, yMo3_valid, yMo3_test = inverse_scale(yMo3_pred_train,yMo3_pred_valid,yMo3_pred_test,yMo3_train,yMo3_valid,yMo3_test,scaler_y_Mo3)
    
    # plot the Mo output values
    plot_outputs(yMo3_train,yMo3_valid,yMo3_test,yMo3_pred_train,yMo3_pred_valid,yMo3_pred_test,idx_Mo3_train,idx_Mo3_valid,idx_Mo3_test,r'Plot for Mo$_3$',log=True, maerun=False,save_path='plots/Mo3.png')
    
    # plot the training history graph
    plot_training_history(history_Mo3_final, title=r'Mo$_3$ Model Training History',save_path='plots/Mo3_model_training.png')
    plot_training_history(combined_Mo3_history, title=r'Mo$_3$ Model Training History',save_path='plots/Mo3_model_combined.png')
    
"""
Undertake the s fits
"""
if run_s:
    # undertake optimisation of the s1 model
    Xs1_test, Xs1_train, Xs1_valid, ys1_test, ys1_train, ys1_valid, idx_s1_test, idx_s1_train, idx_s1_valid = split_data(X_1_scaled,y_s1_scaled,indices_1,test_fraction,validate_to_train_ratio)
    print("Running s1 Optimisation")
    final_model_s1, history_s1_final, combined_s1_history, s1_info = optimise_model(Xs1_train,ys1_train,Xs1_valid,ys1_valid,Xs1_test,ys1_test,space_model,'linear',n_runs,num_optimise,num_initial)
    final_model_s1.save("s1_model.keras")
    
    # Get predictions for each split
    ys1_pred_train = final_model_s1.predict(Xs1_train)
    ys1_pred_valid = final_model_s1.predict(Xs1_valid)
    ys1_pred_test = final_model_s1.predict(Xs1_test)
    ys1_pred_train, ys1_pred_valid, ys1_pred_test, ys1_train, ys1_valid, ys1_test = inverse_scale(ys1_pred_train,ys1_pred_valid,ys1_pred_test,ys1_train,ys1_valid,ys1_test,scaler_y_s1,log=False)
    
    # plot the s output values
    plot_outputs(ys1_train,ys1_valid,ys1_test,ys1_pred_train,ys1_pred_valid,ys1_pred_test,idx_s1_train,idx_s1_valid,idx_s1_test,r'Plot for $\sigma_1$',log=False, maerun=False,save_path='plots/s1.png')
    
    # plot the training history graph
    plot_training_history(history_s1_final, title=r'$\sigma_1$ Model Training History',save_path='plots/s1_model_training.png')
    plot_training_history(combined_s1_history, title=r'$\sigma_1$ Model Training History',save_path='plots/s1_model_combined.png')
    
    # undertake optimisation of the s2 model
    Xs2_test, Xs2_train, Xs2_valid, ys2_test, ys2_train, ys2_valid, idx_s2_test, idx_s2_train, idx_s2_valid = split_data(X_2_scaled,y_s2_scaled,indices_2,test_fraction,validate_to_train_ratio)
    print("Running s2 Optimisation")
    final_model_s2, history_s2_final, combined_s2_history, s2_info = optimise_model(Xs2_train,ys2_train,Xs2_valid,ys2_valid,Xs2_test,ys2_test,space_model,'linear',n_runs,num_optimise,num_initial)
    final_model_s2.save("s2_model.keras")
    
    # Get predictions for each split
    ys2_pred_train = final_model_s2.predict(Xs2_train)
    ys2_pred_valid = final_model_s2.predict(Xs2_valid)
    ys2_pred_test = final_model_s2.predict(Xs2_test)
    ys2_pred_train, ys2_pred_valid, ys2_pred_test, ys2_train, ys2_valid, ys2_test = inverse_scale(ys2_pred_train,ys2_pred_valid,ys2_pred_test,ys2_train,ys2_valid,ys2_test,scaler_y_s2,log=False)
    
    # plot the s output values
    plot_outputs(ys2_train,ys2_valid,ys2_test,ys2_pred_train,ys2_pred_valid,ys2_pred_test,idx_s2_train,idx_s2_valid,idx_s2_test,r'Plot for $\sigma_2$',log=False, maerun=False,save_path='plots/s2.png')
    
    # plot the training history graph
    plot_training_history(history_s2_final, title=r'$\sigma_2$ Model Training History',save_path='plots/s2_model_training.png')
    plot_training_history(combined_s2_history, title=r'$\sigma_2$ Model Training History',save_path='plots/s2_model_combined.png')
    
    # undertake optimisation of the s3 model
    Xs3_test, Xs3_train, Xs3_valid, ys3_test, ys3_train, ys3_valid, idx_s3_test, idx_s3_train, idx_s3_valid = split_data(X_3_scaled,y_s3_scaled,indices_3,test_fraction,validate_to_train_ratio)
    print("Running s3 Optimisation")
    final_model_s3, history_s3_final, combined_s3_history, s3_info = optimise_model(Xs3_train,ys3_train,Xs3_valid,ys3_valid,Xs3_test,ys3_test,space_model,'linear',n_runs,num_optimise,num_initial)
    final_model_s3.save("s3_model.keras")
    
    # Get predictions for each split
    ys3_pred_train = final_model_s3.predict(Xs3_train)
    ys3_pred_valid = final_model_s3.predict(Xs3_valid)
    ys3_pred_test = final_model_s3.predict(Xs3_test)
    ys3_pred_train, ys3_pred_valid, ys3_pred_test, ys3_train, ys3_valid, ys3_test = inverse_scale(ys3_pred_train,ys3_pred_valid,ys3_pred_test,ys3_train,ys3_valid,ys3_test,scaler_y_s3,log=False)
    
    # plot the s output values
    plot_outputs(ys3_train,ys3_valid,ys3_test,ys3_pred_train,ys3_pred_valid,ys3_pred_test,idx_s3_train,idx_s3_valid,idx_s3_test,r'Plot for $\sigma_3$',log=False, maerun=False,save_path='plots/s3.png')
    
    # plot the training history graph
    plot_training_history(history_s3_final, title=r'$\sigma_3$ Model Training History',save_path='plots/s3_model_training.png')
    plot_training_history(combined_s3_history, title=r'$\sigma_3$ Model Training History',save_path='plots/s3_model_combined.png')