name_plot = r'${\rm Spectrum\ name}$'
name_file = 'Spectrum name'
#normalization flux at 1290 A (if necessary)
raw_data = 'pathtorawdata/Spectrumname.txt' #raw data file
#!! make sure that wavelengths are in A and not in log space !!
no_rows_to_skip = 0 #number of rows to skip in the raw data file
nan_vals = '...' #what to label as nan
z_quasar = 0.0 #redshift of the quasar
C = 100 #how many networks in the committee


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

spec, norm, norm_Lya, loglam_raw, flux_raw, f_err_raw, mask = hz.prepare_high_z_data(raw_data=raw_data,z_quasar=z_quasar,\
    skiprows=no_rows_to_skip,na_values=nan_vals,norm_lam=1290,lam_start=1190,lam_stop=2999,bin_size=10,shuffle=5)
loglams = np.array(spec.columns).astype(np.float)

np.savetxt('data/high-z/smoothed/'+str(name_file)+'_reconstructed.txt',np.transpose([10**loglams,spec.values[0,:]]),delimiter=' ',fmt='%1.10f',\
    header='Reconstructed spectrum for '+str(name_file)+' normalized at 1290 A. Normalization constant: '+str(norm)) # This spectrum is normalized wrt 1290 A!

red_orig, blue_orig, loglams = ah.load_reconstructed_file(path_to_file='data/high-z/smoothed/'+str(name_file)+'_reconstructed.txt')

mean_prediction, predictions = ah.apply_QSANNdRA(red_side=red_orig,blue_side=blue_orig,loglams=loglams,path_to_models=path_to_models,name_file=name_file,norm=norm)

ah.plot_predictions(loglam_raw=loglam_raw,flux_raw=flux_raw,f_err_raw=f_err_raw,loglams=loglams,red=red_orig,predictions=predictions,\
    mean_prediction=mean_prediction,norm=norm,norm_Lya=norm_Lya,z_quasar=z_quasar,name_file=name_file,name_plot=name_plot,damping_wing=False,\
        x_HI_mean=None,x_HI_std=None,mask=mask)