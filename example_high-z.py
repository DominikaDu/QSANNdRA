# Input data for the quasar:

name_plot = r'${\rm ULAS\ J1342+0928}$'
name_file = 'ULASJ1342+0928'
#normalization flux at 1290 A (if necessary)
raw_data = 'data/high-z/raw/ulasj1342+0928combined_banados2018.dat' #raw data file
#!! make sure that wavelengths are in A and not in log space !!
no_rows_to_skip = 12 #number of rows to skip in the raw data file
nan_vals = '...' #what to label as nan
z_quasar = 7.5413 #redshift of the quasar
C = 100 #how many networks in the committee

name_plot = r'${\rm ULAS\ J1120+0641}$'
name_file = 'ULASJ1120+0641'
raw_data = 'data/high-z/raw/spectrum_ulas_j1120+0641.dat' #raw data file
#!! make sure that wavelengths are in A and not in log space !!
no_rows_to_skip = 81 #number of rows to skip in the raw data file
nan_vals = '...' #what to label as nan
z_quasar = 7.0851 #redshift of the quasar
C = 100 #how many networks in the committee

name_plot = r'${\rm P183+05}$'
name_file = 'P183+05'
#normalization flux at 1290 A (if necessary)
raw_data = 'data/high-z/raw/P183+05fireyp1v2.scp' #raw data file
#!! make sure that wavelengths are in A and not in log space !!
no_rows_to_skip = 1 #number of rows to skip in the raw data file
nan_vals = 'nan' #what to label as nan
z_quasar = 6.4386 #redshift of the quasar
C = 100 #how many networks in the committee

# Reconstruction interval (observed wavelengths in A):

# for ULAS J1342+0928
lam1_start_obs = 13409.0
lam1_stop_obs = 14554.0
lam2_start_obs = 17850.0
lam2_stop_obs = 19480.0

#for ULAS J1120+0641
#lam1_start_obs =  13421.0
#lam1_stop_obs = 14551.0
#lam2_start_obs = 17786.0
#lam2_stop_obs = 19805.0


#####################################################
import numpy as np
import pandas as pd
from QSmooth import open_calibrate_fits, mask_SDSS, smooth
import preprocessing as pr
import QSANNdRA as Q
import highz_reconstruction as hz
import apply_to_highz as ah
import joblib

path_to_models = 'models/'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

##############################################################################################

# Below is an example use of the red-side reconstruction routines for high-z quasars.

training_spectra = 'data/dataframes/good/training_spectra_cleaned.csv'
training_norms = 'data/dataframes/good/training_spectra_norms_cleaned.csv'

lam1_start = np.divide(lam1_start_obs,(1+z_quasar))
lam1_stop = np.divide(lam1_stop_obs,(1+z_quasar))
lam2_start = np.divide(lam2_start_obs,(1+z_quasar))
lam2_stop = np.divide(lam2_stop_obs,(1+z_quasar))

loglam1_start = np.around(np.log10(lam1_start),decimals=4)
loglam1_stop = np.around(np.log10(lam1_stop),decimals=4)
loglam2_start = np.around(np.log10(lam2_start),decimals=4)
loglam2_stop = np.around(np.log10(lam2_stop),decimals=4)

# Train a NN to predict this wavelength range

spectra, norms = Q.load_training_data(path_to_spectra=training_spectra, path_to_norms=training_norms)
train_x_orig, train_y_orig, test_x_orig, test_y_orig, len_first = hz.high_z_train_test_split(dataframe=spectra,lam1_start=lam1_start,\
    lam1_stop=lam1_stop,lam2_start=lam2_start,lam2_stop=lam2_stop)
scaler_one_x, pca_x, scaler_two_x, train_x, test_x = Q.transform_training_data(train_orig=train_x_orig,test_orig=test_x_orig,n_comp=55)
scaler_one_y, pca_y, scaler_two_y, train_y, test_y = Q.transform_training_data(train_orig=train_y_orig,test_orig=test_y_orig,n_comp=11)

model = hz.train_reconstruction_NN(train_x=train_x,train_y=train_y,path_to_NN='models/high-z/'+str(name_file)+'.h5',load=True)
pred_y = scaler_one_y.inverse_transform(pca_y.inverse_transform(scaler_two_y.inverse_transform(model.predict(test_x))))

err = ((np.abs(pred_y - test_y_orig)/test_y_orig)).mean(axis=0)
std = ((np.abs(pred_y - test_y_orig)/test_y_orig)).std(axis=0)
print('mean epsilon bar: '+str(np.mean(err)))
print('mean sigma epsilon: '+str(np.mean(std)))

# Apply the trained NN to reconstruct the full, smoothed red-side spectrum of the quasar

#name_file = 'P183+05_5'

spec, norm, norm_Lya, loglam_raw, flux_raw, f_err_raw, mask = hz.prepare_high_z_data(raw_data=raw_data,z_quasar=z_quasar,\
    skiprows=no_rows_to_skip,na_values=nan_vals,norm_lam=1290,lam_start=1191.5,lam_stop=2900,bin_size=40,shuffle=20)

print(norm)
x, _, len_first = hz.high_z_split(dataframe=spec,lam1_start=lam1_start,lam1_stop=lam1_stop,lam2_start=lam2_start,lam2_stop=lam2_stop,lam=1290)
x = np.nan_to_num(x)
x_trans = scaler_two_x.transform(pca_x.transform(scaler_one_x.transform(x)))
y_pred = scaler_one_y.inverse_transform(pca_y.inverse_transform(scaler_two_y.inverse_transform(model.predict(x_trans))))

# Concatenate the smoothed red-side spectrum with the smoothed blue-side spectrum and save this array

prediction_first = y_pred[:,:len_first]
prediction_second = y_pred[:,len_first:]
spec.loc[:,str(loglam1_start+0.0001):str(loglam1_stop)] = prediction_first[0,:]
spec.loc[:,str(loglam2_start+0.0001):str(loglam2_stop)] = prediction_second[0,:]
loglams = np.array(spec.columns).astype(np.float)
np.savetxt('data/high-z/raw/'+str(name_file)+'_reconstructed.txt',np.transpose([10**loglams,spec.values[0,:]]),delimiter=' ',fmt='%1.10f',\
    header='Reconstructed spectrum for '+str(name_file)+' normalized at 1290 A. Normalization constant: '+str(norm)) # This spectrum is normalized wrt 1290 A!

# Plot reconstructed spectrum

hz.plot_smoothed_spectrum(loglam_raw=loglam_raw,flux_raw=flux_raw,f_err_raw=f_err_raw,loglams=loglams,spec=spec,norm=norm,\
    name_plot=name_plot,name_file=name_file,mask=mask)

##############################################################################################

# Below is an example use of the blue-side reconstruction routines using QSANNdRA for high-z quasars.

# Apply QSANNdRA to this smoothed, reconstructed spectrum.

#spec, norm, norm_Lya, loglam_raw, flux_raw, f_err_raw, _ = hz.prepare_high_z_data(raw_data=raw_data,z_quasar=z_quasar,\
#    skiprows=no_rows_to_skip,na_values=nan_vals,norm_lam=1290,lam_start=1191.5,lam_stop=2900,bin_size=10,shuffle=5)

red_orig, blue_orig, loglams = ah.load_reconstructed_file(path_to_file='data/high-z/raw/'+str(name_file)+'_reconstructed.txt')

mean_prediction, predictions = ah.apply_QSANNdRA(red_side=red_orig,blue_side=blue_orig,loglams=loglams,path_to_models=path_to_models,name_file=name_file,norm=norm)

# Perform damping wing analysis.

#x_HI_mean, x_HI_std = ah.model_damping_wing(loglam_predictions=loglams,predictions=predictions*norm,loglam_raw=loglam_raw,\
#    flux_raw=flux_raw,f_err_raw=f_err_raw,z_quasar=z_quasar)

ah.plot_predictions(loglam_raw=loglam_raw,flux_raw=flux_raw,f_err_raw=f_err_raw,loglams=loglams,red=red_orig,predictions=predictions,\
    mean_prediction=mean_prediction,norm=norm,norm_Lya=norm_Lya,z_quasar=z_quasar,name_file=name_file,name_plot=name_plot,damping_wing=False,\
        x_HI_mean=None,x_HI_std=None,mask=mask)

# Produce histogram and nearest-neighbour performance figure.