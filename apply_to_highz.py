# -*- coding: utf-8 -*-
##########################################

# Author: Dominika Ďurovčíková (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# D. Ďurovčíková, H. Katz, S. E. I. Bosman, F. B. Davies, J. Devriendt, and A. Slyz, Monthly Notices of the Royal Astronomical Society 493, 4256 (2020).

##########################################

# This module contains routines to apply QSANNdRA to a high-z quasar. For a full description of the
# procedure, please refer to Ďurovčíková et al. 2019 (https://arxiv.org/abs/1912.01050).


# Structure:

# Load prepared high-z data
# Feed into QSANNdRA
# Make a prediction (save as csv, plot)
# Create a nearest-neighbour performance plot
# Create a histogram (how represented this quasar is in SDSS training set)


import numpy as np
import pandas as pd
from scipy import interpolate
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import QSmooth
import preprocessing as pr
import QSANNdRA as Q
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn import linear_model
from scipy.optimize import minimize_scalar



# Blue-side reconstruction routines for high-z quasars.

def import_preprocessing_models(path_to_models):
    scaler_one_red = joblib.load(str(path_to_models)+'scaler_one_red.pkl')
    scaler_one_blue = joblib.load(str(path_to_models)+'scaler_one_blue.pkl')
    pca_red = joblib.load(str(path_to_models)+'pca_red.pkl')
    pca_blue = joblib.load(str(path_to_models)+'pca_blue.pkl')
    scaler_two_red = joblib.load(str(path_to_models)+'scaler_two_red.pkl')
    scaler_two_blue = joblib.load(str(path_to_models)+'scaler_two_blue.pkl')

    return scaler_one_red, scaler_one_blue, pca_red, pca_blue, scaler_two_red, scaler_two_blue

def load_reconstructed_file(path_to_file):
    spec = np.loadtxt(str(path_to_file),delimiter=' ',dtype=float)
    loglams = np.around(np.log10(spec[:,0]),decimals=4)
    flux = spec[:,1]
    spec = pd.DataFrame(data=[flux],columns=np.asarray(loglams,dtype='str'))
    loglams = np.array(spec.columns).astype(np.float)
    red = spec.loc[:,u'3.1106':].values
    blue = spec.loc[:,:u'3.1106'].values

    return red, blue, loglams

def apply_QSANNdRA(red_side,blue_side,loglams,path_to_models,name_file,norm,C=100):
    
    scaler_one_red, scaler_one_blue, pca_red, pca_blue, scaler_two_red, scaler_two_blue = \
        import_preprocessing_models(path_to_models=str(path_to_models)+'preprocessing_models/')

    red = scaler_two_red.transform(pca_red.transform(scaler_one_red.transform(red_side[0,:])))
    
    # Load QSANNdRA and compute blue-side predictions
    predictions = np.empty((np.shape(blue_side)[1],C))
    errs = np.empty((np.shape(blue_side)[1],C))
    s = range(1,C+1)
    for i in range (0,C):
        print(i)
        model = Q.get_NN(seed=s[i])
        model = Q.load_NN_weights(model=model,weights=str(path_to_models)+'QSANNdRA/NN_committee_'+str(i)+'.h5')
        predictions[:,i] = scaler_one_blue.inverse_transform(pca_blue.inverse_transform(scaler_two_blue.inverse_transform(model.predict(red))))
        errors = np.loadtxt(str(path_to_models)+'errors/NN_'+str(i)+'_err_std.csv',delimiter=',')
        errs[:,i] = 1-errors[:,0]

    err_norm = errs.sum(axis=1)
    w = np.divide(errs,err_norm[:,None])
    mean_prediction = np.average(predictions,weights=w,axis=1)
    save_ar = np.concatenate(((np.reshape(10**loglams,(len(loglams),1))[:len(mean_prediction),:]),predictions),axis=1)
    print(np.shape(save_ar))
    np.savetxt('data/high-z/predictions/'+str(name_file)+'_prediction_mean.txt',np.transpose([10**loglams[:len(mean_prediction)],mean_prediction]),delimiter=' ',fmt='%1.10f',\
        header='Mean prediction of QSANNdRA for the spectrum of '+str(name_file)+' normalized at 1290 A. Normalization constant: '+str(norm))
    np.savetxt('data/high-z/predictions/'+str(name_file)+'_predictions.txt',\
            save_ar,delimiter=' ',fmt='%1.10f',\
            header='All predictions of QSANNdRA for the spectrum of '+str(name_file)+' normalized at 1290 A. Normalization constant: '+str(norm))

    return mean_prediction, predictions

def plot_predictions(loglam_raw,flux_raw,f_err_raw,loglams,red,predictions,mean_prediction,norm,norm_Lya,z_quasar,name_file,name_plot,damping_wing=False,x_HI_mean=None,x_HI_std=None,mask=None):
    plt.rc('text',usetex=True)
    font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
    plt.rc('font',**font)
    _, axs = plt.subplots(2,1,sharey=True,figsize=(6.97,3.31))
    axs[0].plot(10**loglam_raw,np.divide(flux_raw,norm_Lya),c=(0.25,0.25,0.25),label=str(name_plot)+r'${\rm \ raw\ data}$')
    axs[1].plot(10**loglam_raw,np.divide(flux_raw,norm_Lya),c=(0.25,0.25,0.25),label=str(name_plot)+' raw data')
    axs[0].set_ylabel(r'${\rm normalized\ flux}$')
    axs[0].set_xlim([10**3.06, 2950])
    axs[0].set_ylim([-0.25,4.0])
    axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
    axs[1].set_ylabel(r'${\rm normalized\ flux}$')
    axs[1].set_xlim([1185, 1300])
    mx = np.max(np.max(np.divide(predictions*norm,norm_Lya)))
    axs[1].set_ylim([-0.02,1.2*mx])
    if len(mask)>0:
        axs[0].vlines(x=(10**loglam_raw[mask]),ymin=-0.02,ymax=1.2*mx,color='lightgray',zorder=0)
        axs[1].vlines(x=(10**loglam_raw[mask]),ymin=-0.02,ymax=1.2*mx,color='lightgray',zorder=0)
    axs[0].plot(10**loglams[:len(predictions[:,0])],np.divide(predictions*norm,norm_Lya),c='m',linewidth = 1,alpha=0.05)
    axs[1].plot(10**loglams[:len(predictions[:,0])],np.divide(predictions*norm,norm_Lya),c='m',linewidth = 1,alpha=0.05)
    if damping_wing:
        lam_tar, _, y, z_s = f(n=x_HI_mean,loglam_prediction=loglams[:len(mean_prediction)],prediction=mean_prediction*norm,loglam=loglam_raw,\
            flux=flux_raw,f_err=f_err_raw,z_quasar=z_quasar,optimize=False)
        axs[0].plot(lam_tar/(1+z_quasar),np.divide(y,norm_Lya),c='dodgerblue',linewidth=1.5,label=r'${\rm damping\ wing\ model\ for\ }$'+r'$\bar{x}_\mathrm{HI}='+np.str(format(x_HI_mean,'.2f'))+'^{+'+np.str(format(x_HI_std,'.2f'))+'}_{-'+np.str(format(x_HI_std,'.2f'))+'}$')
        axs[1].plot(lam_tar/(1+z_quasar),np.divide(y,norm_Lya),c='dodgerblue',linewidth=1.5,label='damping wing model for $x_\mathrm{HI}='+np.str(format(x_HI_mean,'.2f'))+'^{+'+np.str(format(x_HI_std,'.2f'))+'}_{-'+np.str(format(x_HI_std,'.2f'))+'}$')
        axs[0].axvline(x=(1215.67*(1+z_s)/(1+z_quasar)),c='k',linestyle='--',linewidth=1)
        axs[1].axvline(x=(1215.67*(1+z_s)/(1+z_quasar)),c='k',linestyle='--',linewidth=1)
    axs[0].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya),c='mediumvioletred',linewidth=3,label=r'${\rm QSANNdRA: mean\ prediction}$')
    axs[1].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya),c='mediumvioletred',linewidth=3,label='mean prediction')
    NN_err = np.loadtxt('models/errors/QSANNdRA_test_performance.csv',delimiter=',')
    axs[0].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya)*(1+NN_err[:,0]),c='mediumvioletred',linewidth=2,linestyle='dotted',label=r'${\rm QSANNdRA\ uncertainty:\ }$' r'$\overline{\epsilon}$')
    axs[1].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya)*(1+NN_err[:,0]),c='mediumvioletred',linewidth=2,linestyle='dotted',label='mean prediction')
    axs[0].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya)*(1-NN_err[:,0]),c='mediumvioletred',linewidth=2,linestyle='dotted')
    axs[1].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya)*(1-NN_err[:,0]),c='mediumvioletred',linewidth=2,linestyle='dotted')    
    axs[0].axvline(x=(1215.67),c='k',linestyle='--',linewidth=1)
    axs[1].axvline(x=(1215.67),c='k',linestyle='--',linewidth=1)
    axs[0].plot(10**loglams[len(predictions[:,0]):],np.divide(red[0,:]*norm,norm_Lya),c='c')#label=r'${\rm flux\ fit}$')
    axs[1].plot(10**loglams[len(predictions[:,0]):],np.divide(red[0,:]*norm,norm_Lya),c='c',label='flux fit')
    axs[0].plot(10**loglam_raw,np.divide(f_err_raw,norm_Lya),c='r',linewidth=0.5)#,label=r'${\rm raw\ flux\ errors}$')
    axs[1].plot(10**loglam_raw,np.divide(f_err_raw,norm_Lya),c='r',linewidth=0.5,label='raw flux errors')
    axs[0].legend(frameon=False)
    plt.savefig('plots/high-z/'+str(name_file)+'_prediction.png',bbox_inches='tight',dpi=400)


def integ(x):
	y = (x**(9.0/2.0))/(1-x) + (9.0/7.0)*(x**(7.0/2.0)) + (9.0/5.0)*(x**(5.0/2.0)) + \
        3.0*(x**(3.0/2.0)) + 9.0*(x**(1.0/2.0)) - (9.0/2.0)*np.log(np.abs(np.divide((1+(x**(1.0/2.0))),(1-(x**(1.0/2.0))))))
	return(y)

def f(n,loglam_prediction,prediction,loglam,flux,f_err,z_quasar,nz=0.10,z_n=6.0,optimize=True):

	# Fix the cosmology
    h = 0.6766
    Omega_m = 0.3111
    Omega_b = 0.02242/(h**2)	

    # Set target wavelength range for the damping wing analysis
    lam_target = np.linspace(1210*(1+z_quasar),1250*(1+z_quasar),1000)
    lam_Lya = 1215.67

    [lam_smooth, flux_smooth] = QSmooth.smooth((10**loglam)*(1+z_quasar),flux,f_err,mask=[],bin_s=5,shuf=3,Lya=True)
    int_spec = interpolate.interp1d(lam_smooth,flux_smooth)
    norm_Lya = np.float(int_spec(lam_Lya*(1+z_quasar)))
    F_raw = np.divide(int_spec(lam_target),norm_Lya)

	# Based on Jordi Miralda-Escude (1997):
	# Data wavelengths are all calibrated to rest frame

    # Define z_s as the redshift corresponding to the Lya line at the end of quasar's near zone
    # End of the near zone is defined as where the flux drops below nz (e.g. 10%)
    index = len(F_raw[F_raw<nz])
    lam_nz = lam_target[index]
    lam_target = lam_target[index+1:]
    F_raw = F_raw[index+1:]

    z_s = (lam_nz/lam_Lya)-1 #corresponding to the end of the GP trough (i.e. end of quasar near zone)
    z_n = z_n #to be varied

    flux_int = interpolate.interp1d((10**loglam_prediction)*(1+z_quasar),prediction)
    flux_pred = np.divide(flux_int(lam_target),norm_Lya)

    delta_lam = lam_target - lam_Lya*(1+z_s)
    delta = delta_lam/(lam_Lya*(1+z_s))

    f_Lya = 2.47e15 #Hz frequency of Lya in rest frame 
    Lam = 6.25e8 #s^-1 decay constant for Lya resonance
    R_alpha = Lam/(4.0*np.pi*f_Lya)
	
	#from https://arxiv.org/pdf/astro-ph/0512082.pdf
    tau_o = (1.8e5)*(Omega_m**(-1.0/2.0))*(Omega_b*h/0.02)*(((1+z_s)/7.0)**(3.0/2.0))*n

    x_one = (1+z_n)/((1+z_s)*(1+delta))
    x_two = 1/(1+delta)

	#optical depth:
    tau = (tau_o*R_alpha/np.pi)*((1+delta)**(3.0/2.0))*(integ(x_two)-integ(x_one))

    #flux damping:
    F_obs = flux_pred*(np.exp(-tau))

    if optimize:
	    return sum((F_obs-F_raw)**2)
    else:
        return lam_target, F_raw, F_obs*norm_Lya, z_s

def model_damping_wing(loglam_predictions,predictions,loglam_raw,flux_raw,f_err_raw,z_quasar,nz=0.10,z_n=6.0,C=100):
    results = np.empty(C)
    errs = np.empty((np.shape(predictions)[0],C))
    for i in range(0,C):
        print(i)
        result = minimize_scalar(f,args=(loglam_predictions[:np.shape(predictions)[0]],predictions[:,i],\
            loglam_raw,flux_raw,f_err_raw,z_quasar),method='bounded',bounds=(0,1))
        results[i] = result.x
        print('neutral fraction is: '+str(result.x))
        errors = np.loadtxt('models/errors/NN_'+str(i)+'_err_std.csv',delimiter=',')
        errors = errors[:,0]
        errs[:,i] = 1-errors
    
    x_norm = sum(errs.sum(axis=0))
    x_w = np.divide(errs.sum(axis=0),x_norm)
    x_HI_mean = np.average(results,weights=x_w)
    x_HI_std = np.sqrt(np.average((results-x_HI_mean)**2,weights=x_w))
    print(x_HI_mean)
    print(x_HI_std)

    return x_HI_mean, x_HI_std

