from read_data import read_data
from read_fits import read_fits
from plot_trends import plot_trends
from plot_d32_trends import plot_d32_trends
from calc_d32 import calc_d32
from prep_data import prep_data
from remove_zeros import remove_zeros
from scale import scale
from predict_and_inverse import predict_and_inverse
from calc_distribution import calc_distribution

import numpy as np
from build_model import NormalizedReLU, AdjustedSoftmax, ThresholdActivation
from tensorflow.keras.models import load_model

import sys

"""
What are we plotting the variation of "val" against "xaxis"
"""
val = 7 # oil vol frac
xaxis = 2 # viscosity_d

# read in the data
DSD, diameter, properties, labels = read_data('Data_Summary.csv')
fits, fit_labels = read_fits('Data_fits.csv')

# calculate the d32's of the data
d32_exp=np.zeros(len(DSD))
for index in range(len(DSD)):
    d32_exp[index] = calc_d32(DSD[index],diameter)

# Log the required values for the data, viscosity and Modes
properties_fix, fits_fix, indices_f = prep_data(properties,fits)

#Sort the volume fraction parameters into inlet and outlet
X_f = properties_fix[:,1:8].copy()

# if f_i = 0, we do NOT want Mo_i or s_i in that row
X_1, y_Mo1, indices_1 = remove_zeros(properties_fix,fits_fix[:,3],fits_fix[:,1],fits_fix[:,0])
X_1, y_s1, indices_1 = remove_zeros(properties_fix,fits_fix[:,3],fits_fix[:,2],fits_fix[:,0])
X_2, y_Mo2, indices_2 = remove_zeros(properties_fix,fits_fix[:,6],fits_fix[:,4],fits_fix[:,0])
X_2, y_s2, indices_2 = remove_zeros(properties_fix,fits_fix[:,6],fits_fix[:,5],fits_fix[:,0])
X_3, y_Mo3, indices_3 = remove_zeros(properties_fix,fits_fix[:,9],fits_fix[:,7],fits_fix[:,0])
X_3, y_s3, indices_3 = remove_zeros(properties_fix,fits_fix[:,9],fits_fix[:,8],fits_fix[:,0])

# We need the X scalers for the model
scaler_X_f, X_f_scaled = scale(X_f)
scaler_X_1, X_1_scaled = scale(X_1)
scaler_X_2, X_2_scaled = scale(X_2)
scaler_X_3, X_3_scaled = scale(X_3)
del X_1, X_2, X_3, X_f_scaled, X_1_scaled, X_2_scaled, X_3_scaled, indices_f, indices_1, indices_2, indices_3

# y_f does not need scaling as it is between 0 and 1 anyway
# Scale y_Mo's
scaler_y_Mo1, y_Mo1_scaled = scale(y_Mo1)
scaler_y_Mo2, y_Mo2_scaled = scale(y_Mo2)
scaler_y_Mo3, y_Mo3_scaled = scale(y_Mo3)
del y_Mo1, y_Mo2, y_Mo3, y_Mo1_scaled, y_Mo2_scaled, y_Mo3_scaled

# Scale y_s's
scaler_y_s1, y_s1_scaled = scale(y_s1)
scaler_y_s2, y_s2_scaled = scale(y_s2)
scaler_y_s3, y_s3_scaled = scale(y_s3)
del y_s1, y_s2, y_s3, y_s1_scaled, y_s2_scaled, y_s3_scaled

# build d32 output for scaler
y_d32 = np.log(d32_exp).copy()
y_d32 = y_d32.reshape(-1, 1)

#Scale y_d32
scaler_y_d32, y_d32_scaled = scale(y_d32)
del y_d32_scaled, y_d32

# load all the models
final_model_f = load_model(
    "volume_fraction_model.keras",
    custom_objects={
        'NormalizedReLU': NormalizedReLU,
        'AdjustedSoftmax': AdjustedSoftmax,
        'ThresholdActivation': ThresholdActivation
    }
)
final_model_Mo1 = load_model("Mo1_model.keras")
final_model_Mo2 = load_model("Mo2_model.keras")
final_model_Mo3 = load_model("Mo3_model.keras")
final_model_s1 = load_model("s1_model.keras")
final_model_s2 = load_model("s2_model.keras")
final_model_s3 = load_model("s3_model.keras")
final_model_d32 = load_model("d32_model.keras")

# Read in the prediction data
DSD2, diameter2, vis_predict, labels2 = read_data('viscosity_variation_predit.csv')

# Log the required values for the data, viscosity and Modes
properties_vis, fits_holder, indices_holder = prep_data(vis_predict,fits)
X_predict = properties_vis[:,1:8].copy()
del DSD2, diameter2, labels2, fits_holder, indices_holder

# Scale X
X_predict_scale = scaler_X_f.transform(X_predict)

# Volume fraction
yf_pred = predict_and_inverse(final_model_f, X_predict_scale)

# Adjust for Mo and s
X_1 = scaler_X_1.transform(X_predict)
X_2 = scaler_X_2.transform(X_predict)
X_3 = scaler_X_3.transform(X_predict)

# Calculate Mos
yMo1_pred = predict_and_inverse(final_model_Mo1, X_1, scaler_y_Mo1, exp=True)
yMo2_pred = predict_and_inverse(final_model_Mo2, X_2, scaler_y_Mo2, exp=True)
yMo3_pred = predict_and_inverse(final_model_Mo3, X_3, scaler_y_Mo3, exp=True)

# Calculate s's
ys1_pred  = predict_and_inverse(final_model_s1, X_1, scaler_y_s1)
ys2_pred  = predict_and_inverse(final_model_s2, X_2, scaler_y_s2)
ys3_pred  = predict_and_inverse(final_model_s3, X_3, scaler_y_s3)

# Calculate d32's from simple model
yd32_pred = predict_and_inverse(final_model_d32, X_predict_scale, scaler_y_d32, exp=True)

# calculate the predicted distributions and the d32's
d32_pred = np.zeros(len(X_predict))
y_small_cum = np.zeros((len(X_predict),len(diameter)))
y_medium_cum = np.zeros((len(X_predict),len(diameter)))
y_large_cum = np.zeros((len(X_predict),len(diameter)))
y_total_cum = np.zeros((len(X_predict),len(diameter)))
for index in range(len(X_predict)):
    y_small_cum[index], y_medium_cum[index], y_large_cum[index], y_total_cum[index] = [
        arr[:, 0] for arr in calc_distribution(
            diameter,
            [yMo1_pred[index], yMo2_pred[index], yMo3_pred[index]],
            [ys1_pred[index], ys2_pred[index], ys3_pred[index]],
            [yf_pred[index][0], yf_pred[index][1], yf_pred[index][2]]
        )
    ]
    d32_pred[index] = calc_d32(y_total_cum[index],diameter)

# Extract the 7 input features from both arrays
props_features = properties[:, 1:8]
predict_features = vis_predict[:, 1:8]

# Need to remove fits for non-real values, i.e. f=0
for i in range(len(fits)):
    if fits[i,3] == 0:
        fits[i,1:3]=np.nan
    if fits[i,6] == 0:
        fits[i,4:6]=np.nan
    if fits[i,9] == 0:
        fits[i,7:9]=np.nan

# Append the experimental d32s on to the fit array
d32_exp = d32_exp.reshape(-1, 1)
fits = np.concatenate([fits, d32_exp], axis=1)

# Prepare array to hold matched fits
fit_matches = []

# For each row in the prediction features, try to find a match in the training data
for i, row in enumerate(props_features):
    match_indices = np.where(np.all(np.isclose(predict_features, row, atol=1e-8), axis=1))[0]
    if match_indices.size > 0:
        fit_vals = fits[i, 1:]  # skip ID column
        fit_matches.append(fit_vals)
    else:
        fit_matches.append([np.nan] * (fits.shape[1]-1))

fit_matches = np.array(fit_matches)

fit_Mo = fit_matches[:, [0, 3, 6]]
fit_s = fit_matches[:, [1, 4, 7]]
fit_f = fit_matches[:, [2, 5, 8]]
fit_d32 = fit_matches[:, [9]]

unique_fractions = np.unique(vis_predict[:, val]) # this id for the volume fraction

for f_val in unique_fractions:
    
    # Predicted values
    mask = np.isclose(vis_predict[:, val], f_val)
    x = vis_predict[mask, xaxis]  # x-axis values
    
    y_pred_f = yf_pred[mask]
    y_pred_Mo = np.column_stack([yMo1_pred[mask], yMo2_pred[mask], yMo3_pred[mask]])
    y_pred_s = np.column_stack([ys1_pred[mask], ys2_pred[mask], ys3_pred[mask]])
    y_pred_d32 = yd32_pred[mask]
    
    # Need to remove pred for non-real values, i.e. f=0
    for i in range(len(y_pred_f)):
        if y_pred_f[i,0] == 0:
            y_pred_Mo[i,0]=np.nan
            y_pred_s[i,0]=np.nan
        if y_pred_f[i,1] == 0:
            y_pred_Mo[i,1]=np.nan
            y_pred_s[i,1]=np.nan
        if y_pred_f[i,2] == 0:
            y_pred_Mo[i,2]=np.nan
            y_pred_s[i,2]=np.nan
    
    # True values
    mask2 = np.isclose(properties[:, val], f_val)
    x2 = properties[mask2, xaxis]  # x-axis values
    
    y_true_f = fit_f[mask2]
    y_true_Mo = fit_Mo[mask2]
    y_true_s = fit_s[mask2]
    y_true_d32 = fit_d32[mask2]

    # Plot volume fractions
    plot_trends(x, y_pred_f[:, 0], y_pred_f[:, 1], y_pred_f[:, 2],
                x2, y_true_f[:, 0], y_true_f[:, 1], y_true_f[:, 2],
                fr'{labels[val]} = {f_val}',fr'{labels[xaxis]}','Vol fraction',
                save_path=f'plots/f_plot_{f_val}.png')

    # Plot Mo values (log scale)
    plot_trends(x, y_pred_Mo[:, 0], y_pred_Mo[:, 1], y_pred_Mo[:, 2],
                x2, y_true_Mo[:, 0], y_true_Mo[:, 1], y_true_Mo[:, 2],
                fr'{labels[val]} = {f_val}',fr'{labels[xaxis]}',r'Mode / $\mu$m',
                log=True,
                save_path=f'plots/Mo_plot_{f_val}.png')

    # Plot s values
    plot_trends(x, y_pred_s[:, 0], y_pred_s[:, 1], y_pred_s[:, 2],
                x2, y_true_s[:, 0], y_true_s[:, 1], y_true_s[:, 2],
                fr'{labels[val]} = {f_val}',fr'{labels[xaxis]}',r'Standard deviation',
                save_path=f'plots/s_plot_{f_val}.png')
    
    # Plot d32 values (log scale)
    plot_d32_trends(x, y_pred_d32[:,0],
                    x2, y_true_d32[:,0],
                    fr'{labels[val]} = {f_val}',fr'{labels[xaxis]}',r'$d_{32}$ / $\mu$m',
                    log=True,
                    save_path=f'plots/d32_plot_{f_val}.png')