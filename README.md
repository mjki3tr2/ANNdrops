# ANNdrops
ANN models for droplet size modelling

run.py is the main file for the ANN fitting. The data is normalised and then a test set is selected. The ANN model parameters are then optimised using K-fold. This is undertaken for f, Mo1 to Mo3, and s1 to s3.

run_DSD_plots.py takes the models built from run.py and then uses them to compare the actual and predicted d32s and also the DSDs.

run_model_variations.py takes the models built from run.py and a set of input parameters and plots the variation of the model with the actual values.

run_d32_model.py is the same as run.py, but only fits the ANN model for the d32s directly.

run_full.py is the same as run.py, but takes the whole distribution and fits each point in it directly with an ANN.

All other files are functions called by these master files.
