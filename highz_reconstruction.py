# -*- coding: utf-8 -*-
##########################################

# Author: Dominika Ďurovčíková (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# Ďurovčíková, D., Katz, H., Bosman, S.E.I., Davies, F.B., Devriendt, J. and Slyz, A., 2019.
# Reionization history constraints from neural network based predictions of high-redshift quasar continua.
# arXiv preprint arXiv:1912.01050.

##########################################

# This module contains routines to process high-z quasar data for the application fo QSANNdRA. 
# For a full description of the procedure, please refer to Ďurovčíková et al. 2019 (https://arxiv.org/abs/1912.01050).

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

# Red-side reconstruction routines for high-z quasars.

def high_z_train_test_split(dataframe,lam1_start,lam1_stop,lam2_start,lam2_stop,ratio=0.8,lam=1290):
    # The input has to be a pandas dataframe with labelled columns

    train = dataframe.iloc[:(np.int(ratio*len(dataframe)))]
    test = dataframe.iloc[(np.int(ratio*len(dataframe))):]

    loglam1_start = np.around(np.log10(lam1_start),decimals=4)
    loglam1_stop = np.around(np.log10(lam1_stop),decimals=4)
    loglam2_start = np.around(np.log10(lam2_start),decimals=4)
    loglam2_stop = np.around(np.log10(lam2_stop),decimals=4)
    loglam = np.around(np.log10(lam),decimals=4)

    train_x = np.concatenate((train.loc[:,str(loglam):str(loglam1_start)].values,train.loc[:,str(loglam1_stop+0.0001):str(loglam2_start)].values,train.loc[:,str(loglam2_stop+0.0001):].values),axis=1)
    train_y = np.concatenate((train.loc[:,str(loglam1_start+0.0001):str(loglam1_stop)].values,train.loc[:,str(loglam2_start+0.0001):str(loglam2_stop)].values),axis=1)
    test_x = np.concatenate((test.loc[:,str(loglam):str(loglam1_start)].values,test.loc[:,str(loglam1_stop+0.0001):str(loglam2_start)].values,test.loc[:,str(loglam2_stop+0.0001):].values),axis=1)
    test_y = np.concatenate((test.loc[:,str(loglam1_start+0.0001):str(loglam1_stop)].values,test.loc[:,str(loglam2_start+0.0001):str(loglam2_stop)].values),axis=1)

    len_first = np.shape(test.loc[:,str(loglam1_start+0.0001):str(loglam1_stop)].values)[1]

    return train_x, train_y, test_x, test_y, len_first

def high_z_split(dataframe,lam1_start,lam1_stop,lam2_start,lam2_stop,lam=1290):
    # The input has to be a pandas dataframe with labelled columns

    loglam1_start = np.around(np.log10(lam1_start),decimals=4)
    loglam1_stop = np.around(np.log10(lam1_stop),decimals=4)
    loglam2_start = np.around(np.log10(lam2_start),decimals=4)
    loglam2_stop = np.around(np.log10(lam2_stop),decimals=4)
    loglam = np.around(np.log10(lam),decimals=4)

    x = np.concatenate((dataframe.loc[:,str(loglam):str(loglam1_start)].values,dataframe.loc[:,str(loglam1_stop+0.0001):str(loglam2_start)].values,dataframe.loc[:,str(loglam2_stop+0.0001):].values),axis=1)
    y = np.concatenate((dataframe.loc[:,str(loglam1_start+0.0001):str(loglam1_stop)].values,dataframe.loc[:,str(loglam2_start+0.0001):str(loglam2_stop)].values),axis=1)

    len_first = np.shape(dataframe.loc[:,str(loglam1_start+0.0001):str(loglam1_stop)].values)[1]

    return x, y, len_first

def train_reconstruction_NN(train_x,train_y,load=False,path_to_NN=None):
    model = Sequential()
    model.add(Dense(20, input_dim=55, kernel_initializer='normal',activation='elu'))
    model.add(Dense(11, kernel_initializer='normal')) #output layer
    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mae'])
    if load:
        model = Q.load_NN_weights(model=model,weights=str(path_to_NN))
    else:
        history, model = Q.train_NN(model=model,train_red=train_x,train_blue=train_y,checkpointer_path=str(path_to_NN),epochs=400,batch_size=800)

    return model, history

def prepare_high_z_data(raw_data,z_quasar,skiprows,na_values,norm_lam=1290,lam_start=1191.5,lam_stop=2900):
    
    Lya_loglam = np.around(np.log10(1215.0),decimals=4)
    norm_loglam = np.around(np.log10(norm_lam),decimals=4)
    loglam_start = np.around(np.log10(lam_start),decimals=4)
    loglam_stop = np.around(np.log10(lam_stop),decimals=4)    
    loglam_target = np.around(np.arange(loglam_start,loglam_stop-0.0001,0.0001),decimals=4)

    data = pd.read_csv(raw_data,skiprows=skiprows,sep='\s+',dtype=float,na_values=na_values)
    loglam = np.around(np.log10(np.divide(data.values[:,0],1+z_quasar)),decimals=4)
    flux = data.values[:,1]
    f_err = data.values[:,2]

    # Create a mask to ignore nan values

    mask = np.zeros(len(loglam), dtype=np.int)
    for i in range(0,len(loglam)):
        if np.isnan(f_err[i]):
            mask[i] = 1
    mask_bool = mask==0

    [loglam_smooth, flux_smooth] = QSmooth.smooth(loglam,flux,f_err,mask=[],bin_s=30,shuf=15,Lya=False)
    int_spec = interpolate.interp1d(loglam_smooth,flux_smooth,fill_value='extrapolate')
    norm = np.float(int_spec(norm_loglam))
    norm_Lya = np.float(int_spec(Lya_loglam))
    flux_int = np.divide(int_spec(loglam_target),norm)

    df_norm = pd.DataFrame(data=[flux_int],columns=np.asarray(loglam_target,dtype='str'))
    df_norm = df_norm.drop(columns=str(norm_loglam))

    return df_norm, norm, norm_Lya, loglam, flux, f_err

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
    spec = np.loadtxt(str(path_to_file),delimiter=',',dtype=float)
    loglams = spec[0,:]
    flux = spec[1,:]
    spec = pd.DataFrame(data=[flux],columns=np.asarray(loglams,dtype='str'))
    loglams = np.array(spec.columns).astype(np.float)
    red = spec.loc[:,u'3.1106':].values
    blue = spec.loc[:,:u'3.1106'].values

    return red, blue, loglams

def apply_QSANNdRA(red_side,blue_side,loglams,path_to_models,name_file,C=100):
    
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
    np.savetxt('data/high-z/predictions/'+str(name_file)+'_prediction_mean.csv',np.transpose([loglams[:len(mean_prediction)],mean_prediction]),delimiter=',')
    np.savetxt('data/high-z/predictions/'+str(name_file)+'_predictions.csv',\
        np.transpose(np.concatenate((np.reshape(loglams,(len(loglams),1))[:len(mean_prediction),:],predictions),axis=1)),delimiter=',')

    return mean_prediction, predictions

def plot_predictions(loglam_raw,flux_raw,f_err_raw,loglams,red,predictions,mean_prediction,norm,norm_Lya,name_file,name_plot):
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
    axs[1].set_ylim([-0.02,4.0])
    axs[0].plot(10**loglams[:len(predictions[:,0])],np.divide(predictions*norm,norm_Lya),c='m',linewidth = 1,alpha=0.05)
    axs[1].plot(10**loglams[:len(predictions[:,0])],np.divide(predictions*norm,norm_Lya),c='m',linewidth = 1,alpha=0.05)
    axs[0].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya),c='mediumvioletred',linewidth=3,label=r'${\rm QSANNdRA: mean\ prediction}$')
    axs[1].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya),c='mediumvioletred',linewidth=3,label='mean prediction')
    axs[0].axvline(x=(1215.67),c='k',linestyle='--',linewidth=1)
    axs[1].axvline(x=(1215.67),c='k',linestyle='--',linewidth=1)
    axs[0].plot(10**loglams[len(predictions[:,0]):],np.divide(red[0,:]*norm,norm_Lya),c='c')#label=r'${\rm flux\ fit}$')
    axs[1].plot(10**loglams[len(predictions[:,0]):],np.divide(red[0,:]*norm,norm_Lya),c='c',label='flux fit')
    axs[0].plot(loglam_raw,np.divide(f_err_raw,norm_Lya),c='r',linewidth=0.5)#,label=r'${\rm raw\ flux\ errors}$')
    axs[1].plot(loglam_raw,np.divide(f_err_raw,norm_Lya),c='r',linewidth=0.5,label='raw flux errors')
    axs[0].legend(frameon=False)
    plt.savefig('plots/high-z/'+str(name_file)+'_prediction.png',bbox_inches='tight',dpi=400)






def integ(x):
	y = (x**(9.0/2.0))/(1-x) + (9.0/7.0)*(x**(7.0/2.0)) + (9.0/5.0)*(x**(5.0/2.0)) + \
        3.0*(x**(3.0/2.0)) + 9.0*(x**(1.0/2.0)) - (9.0/2.0)*np.log(np.abs(np.divide((1+(x**(1.0/2.0))),(1-(x**(1.0/2.0))))))
	return(y)

def f(prediction_file,norm,z_quasar,raw_data,no_rows_to_skip,nan_vals,optimize=False):

	#fix the cosmology
    h = 0.6766
    Omega_m = 0.3111
    Omega_b = 0.02242/(h**2)	

    lam_target = np.linspace(1210*(1+z_quasar),1250*(1+z_quasar),1000)

    data = pd.read_table(raw_data,skiprows=no_rows_to_skip,sep='\s+',dtype=float,na_values=nan_vals)
    lam_raw = np.around(data.values[:,0],decimals=4)
    flux_raw = data.values[:,1]
    f_err = data.values[:,2]

	#compute F_raw
    [xx,yy]=smooth(lam_raw,flux_raw,f_err)
    F_int = interp1d(xx,yy)
    norm_Lya = np.float(F_int(1215.67*(1+z_quasar)))
    F_raw = np.divide(F_int(lam_target),norm_Lya)
    flux_raw = np.copy(np.divide(flux_raw,norm_Lya))

	#based on Jordi Miralda-Escude (1997):
	#data wavelengths are all calibrated to rest frame

    #define z_s as the redshift corresponding to the Lya line at the end of quasar's near zone
    #end of the near zone is defined as where the flux drops below nz (e.g. 10%)
    index = len(F_raw[F_raw<nz])
    lam_nz = lam_target[index]
    lam_target = lam_target[index+1:]
    F_raw = np.divide(F_int(lam_target),norm_Lya)

    z_s = (lam_nz/1215.67)-1 #corresponding to the end of the GP trough (i.e. end of quasar near zone)
    z_n = 6.0 #to be varied

    data_pred = np.loadtxt(str(name_file)+'_prediction_'+str(i)+'.csv',delimiter=',',dtype=float)
    lam_pred = (10**(data_pred[:,0]))*(1+z_quasar)
    lam_pred_copy = np.copy(lam_pred)
    flux_pred = np.divide(data_pred[:,1]*norm_df,norm_Lya)
    flux_pred_copy = np.copy(flux_pred)
    flux_int = interp1d(lam_pred_copy,flux_pred_copy)
    flux_pred = flux_int(lam_target)

    lam_Lya = 1215.67 #A Lya rest frame wavelength
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
        return lam_target, F_raw, F_obs, z_s