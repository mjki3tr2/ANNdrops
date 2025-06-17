from read_data import read_data
from read_fits import read_fits
from plot_DSD_comp import plot_DSD_comp
from calc_d32 import calc_d32
from prep_data import prep_data
from remove_zeros import remove_zeros
from scale import scale
from predict_and_inverse import predict_and_inverse
from calc_distribution import calc_distribution
from plot_xy import plot_xy
from visualize_model import visualize_model

import numpy as np
from build_model import NormalizedReLU, AdjustedSoftmax, ThresholdActivation
from tensorflow.keras.models import load_model

import sys

input_labels=[r'$N$',r'$\ln\mu_d$',r'$\rho_d$',r'$\ln\mu_c$',r'$\rho_c$',r'$\sigma$',r'$\phi$']

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
del X_1, X_2, X_3, X_1_scaled, X_2_scaled, X_3_scaled

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

# load all the models
final_model_f = load_model(
    "volume_fraction_model.keras",
    custom_objects={
        'NormalizedReLU': NormalizedReLU,
        'AdjustedSoftmax': AdjustedSoftmax,
        'ThresholdActivation': ThresholdActivation
    }
)
f_network = visualize_model(final_model_f,filename='plots/f_network.png',input_labels=input_labels,output_labels=[r'$f_1$',r'$f_2$',r'$f_3$'])

final_model_Mo1 = load_model("Mo1_model.keras")
Mo1_network = visualize_model(final_model_Mo1,filename='plots/Mo1_network.png',input_labels=input_labels,output_labels=[r'$\ln Mo_1$'])
final_model_Mo2 = load_model("Mo2_model.keras")
Mo2_network = visualize_model(final_model_Mo2,filename='plots/Mo2_network.png',input_labels=input_labels,output_labels=[r'$\ln Mo_2$'])
final_model_Mo3 = load_model("Mo3_model.keras")
Mo3_network = visualize_model(final_model_Mo3,filename='plots/Mo3_network.png',input_labels=input_labels,output_labels=[r'$\ln Mo_3$'])
final_model_s1 = load_model("s1_model.keras")
s1_network = visualize_model(final_model_s1,filename='plots/s1_network.png',input_labels=input_labels,output_labels=[r'$s_1$'])
final_model_s2 = load_model("s2_model.keras")
s2_network = visualize_model(final_model_s2,filename='plots/s2_network.png',input_labels=input_labels,output_labels=[r'$s_2$'])
final_model_s3 = load_model("s3_model.keras")
s3_network = visualize_model(final_model_s3,filename='plots/s3_network.png',input_labels=input_labels,output_labels=[r'$s_3$'])

# calculate a prediction of all the data
yf_pred = predict_and_inverse(final_model_f, X_f_scaled)

X_1 = scaler_X_1.transform(X_f)
X_2 = scaler_X_2.transform(X_f)
X_3 = scaler_X_3.transform(X_f)

yMo1_pred = predict_and_inverse(final_model_Mo1, X_1, scaler_y_Mo1, exp=True)
yMo2_pred = predict_and_inverse(final_model_Mo2, X_2, scaler_y_Mo2, exp=True)
yMo3_pred = predict_and_inverse(final_model_Mo3, X_3, scaler_y_Mo3, exp=True)

ys1_pred  = predict_and_inverse(final_model_s1, X_1, scaler_y_s1)
ys2_pred  = predict_and_inverse(final_model_s2, X_2, scaler_y_s2)
ys3_pred  = predict_and_inverse(final_model_s3, X_3, scaler_y_s3)

# calculate the predicted distributions and the d32's
d32_pred = np.zeros(len(DSD))
y_small_cum = np.zeros_like(DSD)
y_medium_cum = np.zeros_like(DSD)
y_large_cum = np.zeros_like(DSD)
y_total_cum = np.zeros_like(DSD)
for index in range(len(DSD)):
    y_small_cum[index], y_medium_cum[index], y_large_cum[index], y_total_cum[index] = [
        arr[:, 0] for arr in calc_distribution(
            diameter,
            [yMo1_pred[index], yMo2_pred[index], yMo3_pred[index]],
            [ys1_pred[index], ys2_pred[index], ys3_pred[index]],
            [yf_pred[index][0], yf_pred[index][1], yf_pred[index][2]]
        )
    ]
    d32_pred[index] = calc_d32(y_total_cum[index],diameter)

plot_xy(d32_exp,d32_pred,title_str=r'Plot for $d_{32}$',save_path='plots/d32.png')

#plot DSDs
for i, index in enumerate(indices_f):
    index=int(index.item())
    savepath=f'DSD_plots/{index:03d}.png'
    plot_DSD_comp(diameter,DSD,labels,properties,y_small_cum,y_medium_cum,y_large_cum,y_total_cum,i,save_path=savepath)
