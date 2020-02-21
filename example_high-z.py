# Input data for the quasar:

name_plot = r'${\rm P183+05}$'
name_file = 'P183+05'
norm_df = 6.209792262755939e-18
#normalization flux at 1290 A (if necessary)
raw_data = 'data/high-z/P183+05fireyp1v2.scp' #raw data file
#!! make sure that wavelengths are in A and not in log space !!
no_rows_to_skip = 12 #number of rows to skip in the raw data file
nan_vals = '...' #what to label as nan
z_quasar = 6.4386 #redshift of the quasar
#if reconstruction of any parts of the spectrum was necessary
reconstructed_file = 'data/high-z/ULASJ1342+0928_reconstructed.csv'
C = 100 #how many networks in the committee
nz = 0.10

# Reconstruction interval (observed wavelengths in A):

lam1_start_obs = 13409.0
lam1_stop_obs = 14554.0
lam2_start_obs = 17937.0
lam2_stop_obs = 19483.0



#####################################################
import numpy as np
import pandas as pd
from QSmooth import open_calibrate_fits, mask_SDSS, smooth
import preprocessing as pr
import QSANNdRA as Q
import highz_reconstruction as hz



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

training_spectra = 'data/dataframes/good/training_spectra_cleaned.csv'
training_norms = 'data/dataframes/good/training_spectra_norms_cleaned.csv'
path_to_models = 'models/'

lam1_start = np.divide(lam1_start_obs,(1+z_quasar))
lam1_stop = np.divide(lam1_stop_obs,(1+z_quasar))
lam2_start = np.divide(lam2_start_obs,(1+z_quasar))
lam2_stop = np.divide(lam2_stop_obs,(1+z_quasar))

# Train a NN to predict this wavelength range

spectra, norms = Q.load_training_data(path_to_spectra=training_spectra, path_to_norms=training_norms)
train_x_orig, train_y_orig, test_x_orig, test_y_orig, len_first = hz.high_z_train_test_split(dataframe=spectra,lam1_start=lam1_start,\
    lam1_stop=lam1_stop,lam2_start=lam2_start,lam2_stop=lam2_stop)
scaler_one_x, pca_x, scaler_two_x, train_x, test_x = Q.transform_training_data(train_orig=train_x_orig,test_orig=test_x_orig,n_comp=55)
scaler_one_y, pca_y, scaler_two_y, train_y, test_y = Q.transform_training_data(train_orig=train_y_orig,test_orig=test_y_orig,n_comp=11)

model, history = hz.train_reconstruction_NN(train_x=train_x,train_y=train_y,path_to_NN='models/high-z/ULASJ1342+0928.h5')
pred_y = scaler_one_y.inverse_transform(pca_y.inverse_transform(scaler_two_y.inverse_transform(model.predict(test_x))))

err = ((np.abs(pred_y - test_y_orig)/test_y_orig)).mean(axis=0)
std = ((np.abs(pred_y - test_y_orig)/test_y_orig)).std(axis=0)
print(np.mean(err))
print(np.mean(std))

# Apply the trained NN to reconstruct the full, smoothed red-side spectrum of the quasar

spec = hz.prepare_high_z_data(raw_data=raw_data,z_quasar=z_quasar,skiprows=no_rows_to_skip,na_values=nan_vals,norm_lam=1290,lam_start=1191.5,lam_stop=2900)
x, _, len_first = hz.high_z_split(dataframe=spec,lam1_start=lam1_start,lam1_stop=lam1_stop,lam2_start=lam2_start,lam2_stop=lam2_stop,lam=1290)
x_trans = scaler_two_x.transform(pca_x.transform(scaler_one_x.transform(x)))
y_pred = scaler_one_y.inverse_transform(pca_y.inverse_transform(scaler_two_y.inverse_transform(model.predict(x_trans))))

# Concatenate the smoothed red-side spectrum with the smoothed blue-side spectrum and save this array

loglam1_start = np.around(np.log10(lam1_start),decimals=4)
loglam1_stop = np.around(np.log10(lam1_stop),decimals=4)
loglam2_start = np.around(np.log10(lam2_start),decimals=4)
loglam2_stop = np.around(np.log10(lam2_stop),decimals=4)

prediction_first = y_pred[:,:len_first]
prediction_second = y_pred[:,len_first:]
spec.loc[:,str(loglam1_start+0.0001):str(loglam1_stop)] = prediction_first[0,:]
spec.loc[:,str(loglam2_start+0.0001):str(loglam2_stop)] = prediction_second[0,:]
loglams_final = np.array(spec.columns).astype(np.float)
np.savetxt('data/high-z/'+str(name_file)+'_reconstructed.csv',[loglams_final,spec.values[0,:]],delimiter=',')

# Apply QSANNdRA to this smoothed, reconstructed spectrum.



# Produce histogram and nearest-neighbour performance figure.