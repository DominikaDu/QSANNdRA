# -*- coding: utf-8 -*-
##########################################

# Author: Dominika Ďurovčíková (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# Ďurovčíková, D., Katz, H., Bosman, S.E.I., Davies, F.B., Devriendt, J. and Slyz, A., 2019.
# Reionization history constraints from neural network based predictions of high-redshift quasar continua.
# arXiv preprint arXiv:1912.01050.

##########################################

# This module contains routines to build and train QSANNdRA. For a full description of the
# procedure, please refer to Ďurovčíková et al. 2019 (https://arxiv.org/abs/1912.01050).


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cmasher as cmr
plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
import pandas as pd
from astropy.io import fits
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, re
os.environ['KERAS_BACKEND'] = 'theano'
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import initializers
import preprocessing as pr
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

def save_model(model,target_path):
    joblib.dump(model,str(target_path))

def load_training_data(path_to_spectra,path_to_norms):
    spectra = pd.read_csv(path_to_spectra)
    norms = pd.read_csv(path_to_norms)
    spectra = spectra.drop(spectra.columns[0],axis=1)
    norms = norms.drop(norms.columns[0],axis=1)

    return spectra, norms

def extract_lambdas(dataframe):
    loglams = np.array(dataframe.columns).astype(np.float)
    lambdas = 10**loglams

    return lambdas

def train_test_split(dataframe,ratio=0.8,lam=1290):
    # The input has to be a pandas dataframe with labelled columns

    train = dataframe.iloc[:(np.int(ratio*len(dataframe)))]
    test = dataframe.iloc[(np.int(ratio*len(dataframe))):]

    loglam = np.around(np.log10(lam),decimals=4)
    train_blue = train.loc[:,:str(loglam)].values
    train_red = train.loc[:,str(loglam):].values
    test_blue = test.loc[:,:str(loglam)].values
    test_red = test.loc[:,str(loglam):].values # This division is OK, as 'loglam' column is dropped (ie no overlap in values)
    print(np.shape(train_blue))
    print(np.shape(train_red))

    return train_red, train_blue, test_red, test_blue

def standardize(train,test,save=False,target_path=None):
    scaler = StandardScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    if save:
        save_model(scaler,target_path)

    return scaler, train_scaled, test_scaled

def perform_PCA(train,test,n_components,save=False,target_path=None):
    pca = PCA(n_components=n_components)
    pca.fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)
    if save:
        save_model(pca,target_path)

    return pca, train_pca, test_pca

def transform_training_data(train_orig,test_orig,n_comp,save=False,target_path=None,name=None):
    scaler_one, train, test = standardize(train=train_orig,test=test_orig,save=save,target_path=str(target_path)+'scaler_one_'+str(name)+'.pkl')
    pca, train, test = perform_PCA(train=train,test=test,n_components=n_comp,save=save,target_path=str(target_path)+'pca_'+str(name)+'.pkl')
    scaler_two, train, test = standardize(train=train,test=test,save=save,target_path=str(target_path)+'scaler_two_'+str(name)+'.pkl')

    return scaler_one, pca, scaler_two, train, test

def prepare_training_data(training_spectra,training_norms,models_save=False,path_to_models=None):
    # Loads and splits training data into train and test sets and performs a sequence of standardization, PCA, standardization.

    spectra, norms = load_training_data(path_to_spectra=training_spectra,path_to_norms=training_norms)
    lam = extract_lambdas(dataframe=spectra)
    train_red_orig, train_blue_orig, test_red_orig, test_blue_orig = \
        train_test_split(dataframe=spectra,ratio=0.8,lam=1290)
    scaler_red_one, pca_red, scaler_red_two, train_red, test_red = \
        transform_training_data(train_orig=train_red_orig,test_orig=test_red_orig,n_comp=63,save=models_save,target_path=path_to_models,name='red')
    scaler_blue_one, pca_blue, scaler_blue_two, train_blue, test_blue = \
        transform_training_data(train_orig=train_blue_orig,test_orig=test_blue_orig,n_comp=36,save=models_save,target_path=path_to_models,name='blue')

    return scaler_red_one, pca_red, scaler_red_two, train_red, test_red, scaler_blue_one, pca_blue, scaler_blue_two, train_blue, test_blue, lam

def get_NN(red_comp=63,n_two=40,n_three=40,blue_comp=36,seed=0,act='elu',optimizer='adam'):
    model = Sequential()
    model.add(Dense(n_two, input_dim=red_comp, kernel_initializer=initializers.RandomNormal(seed=seed),bias_initializer=initializers.RandomNormal(seed=seed),activation=act))
    model.add(Dense(n_three, kernel_initializer=initializers.RandomNormal(seed=seed),bias_initializer=initializers.RandomNormal(seed=seed),activation=act))
    model.add(Dense(blue_comp, kernel_initializer=initializers.RandomNormal(seed=seed),bias_initializer=initializers.RandomNormal(seed=seed)))
    model.compile(loss='mean_absolute_error',optimizer=optimizer,metrics=['mae'])
    return model

def load_NN_weights(model,weights):
    model.load_weights(weights)

    return model

def train_NN(model,train_red,train_blue,checkpointer_path,epochs=80,batch_size=500,validation_split=0.2):
    checkpointer = ModelCheckpoint(filepath=checkpointer_path,verbose=1,save_best_only=False,monitor='val_loss',mode='min',period=5)
    history = model.fit(train_red, train_blue, epochs=epochs, batch_size=batch_size,verbose=1, shuffle=True,validation_split=validation_split,callbacks=[checkpointer])
    
    return history, model 

def get_QSANNdRA(training_spectra,training_norms,models_save=False,path_to_models=None,C=100,load=False,save_errs=False):
    # Load, split and prepare training data
    spectra, norms = load_training_data(path_to_spectra=training_spectra,path_to_norms=training_norms)
    lam = extract_lambdas(dataframe=spectra)
    train_red_orig, train_blue_orig, test_red_orig, test_blue_orig = \
        train_test_split(dataframe=spectra,ratio=0.8,lam=1290)
    scaler_one_red, pca_red, scaler_two_red, train_red, test_red = \
        transform_training_data(train_orig=train_red_orig,test_orig=test_red_orig,n_comp=63,save=models_save,target_path=str(path_to_models)+'preprocessing_models/',name='red')
    scaler_one_blue, pca_blue, scaler_two_blue, train_blue, test_blue = \
        transform_training_data(train_orig=train_blue_orig,test_orig=test_blue_orig,n_comp=36,save=models_save,target_path=str(path_to_models)+'preprocessing_models/',name='blue')    
    
    # Construct and train/load the full committee
    predictions_test = np.empty((np.shape(test_blue_orig)[0],np.shape(test_blue_orig)[1],C))
    errs = np.empty((np.shape(test_blue_orig)[1],C))
    predictions_train = np.empty((np.shape(train_blue_orig)[0],np.shape(train_blue_orig)[1],C))
    errs_train = np.empty((np.shape(train_blue_orig)[1],C))
    s = range(1,C+1)
    for i in range (0,C):
        model = get_NN(seed=s[i])
        if load:
            model = load_NN_weights(model=model,weights=str(path_to_models)+'QSANNdRA/NN_'+str(i)+'.h5')
        else:
            history, model = train_NN(model=model,train_red=train_red,train_blue=train_blue,checkpointer_path=str(path_to_models)+'QSANNdRA/NN_'+str(i)+'.h5')
        
        # Inverse transform and evaluate errors
        pred_test = scaler_one_blue.inverse_transform(pca_blue.inverse_transform(scaler_two_blue.inverse_transform(model.predict(test_red))))
        predictions_test[:,:,i] = pred_test

        pred_train = scaler_one_blue.inverse_transform(pca_blue.inverse_transform(scaler_two_blue.inverse_transform(model.predict(train_red))))
        predictions_train[:,:,i] = pred_train

        err = ((np.abs(pred_test - test_blue_orig)/test_blue_orig)).mean(axis=0) # Calculate mean absolute relative error for each wavelength
        std = ((np.abs(pred_test - test_blue_orig)/test_blue_orig)).std(axis=0)

        err_train = ((np.abs(pred_train - train_blue_orig)/train_blue_orig)).mean(axis=0) # Calculate mean absolute relative error for each wavelength
        std_train = ((np.abs(pred_train - train_blue_orig)/train_blue_orig)).std(axis=0)

        if save_errs:
            np.savetxt(str(path_to_models)+'errors/NN_'+str(i)+'_err_std.csv',np.transpose([err,std]),delimiter=',')
        
        errs[:,i] = 1-err
        errs_train[:,i] = 1-err_train

    # Calculate weighted mean of predictions
    err_norm = errs.sum(axis=1)
    w = np.divide(errs,err_norm[:,None])
    w_test = np.repeat(w[np.newaxis,:, :], len(predictions_test), axis=0)
    mean_preds_test = np.average(predictions_test,weights=w_test,axis=2)

    err_mean = ((np.abs(mean_preds_test - test_blue_orig)/test_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength
    std_mean = ((np.abs(mean_preds_test - test_blue_orig)/test_blue_orig)).std(axis=0) #calculate std squared relative error for each wavelength
    np.savetxt(str(path_to_models)+'errors/QSANNdRA_test_performance.csv',np.transpose([err_mean,std_mean]),delimiter=',')
    print('test set err and std:')
    print(np.mean(err_mean))
    print(np.mean(std_mean))

    err_norm = errs_train.sum(axis=1)
    w = np.divide(errs_train,err_norm[:,None])
    w_train = np.repeat(w[np.newaxis,:, :], len(predictions_train), axis=0)
    mean_preds_train = np.average(predictions_train,weights=w_train,axis=2)

    err_mean = ((np.abs(mean_preds_train - train_blue_orig)/train_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength
    std_mean = ((np.abs(mean_preds_train - train_blue_orig)/train_blue_orig)).std(axis=0) #calculate std squared relative error for each wavelength
    np.savetxt(str(path_to_models)+'errors/QSANNdRA_train_performance.csv',np.transpose([err_mean,std_mean]),delimiter=',')
    print('train set err and std:')
    print(np.mean(err_mean))
    print(np.mean(std_mean))

    return predictions_test, mean_preds_test

def plot_QSANNdRA_performance(path_to_models,training_spectra,training_norms,lam_out):
    test = pd.read_csv(str(path_to_models)+'errors/QSANNdRA_test_performance.csv',header=None)
    testerr = test.iloc[:,0]
    teststd = test.iloc[:,1]
    train = pd.read_csv(str(path_to_models)+'errors/QSANNdRA_train_performance.csv',header=None)
    trainerr = train.iloc[:,0]
    trainstd = train.iloc[:,1]
    spectra, _ = load_training_data(path_to_spectra=training_spectra,path_to_norms=training_norms)
    lam = extract_lambdas(dataframe=spectra)
    plt.figure(figsize=(3.31,3.31))
    plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
    plt.ylabel(r'${\rm fractional\ prediction\ error}$')
    plt.xlim([lam_out[0],lam_out[1]])
    plt.ylim([0.0,0.15])
    plt.plot(lam[:len(trainerr)],trainerr,c=(0.25,0.25,0.25),label=r'${\rm mean\ train\ error}$')
    plt.plot(lam[:len(trainstd)],trainstd,c=(0.25,0.25,0.25),linestyle='dashed',label=r'${\rm train\ \sigma }$')
    plt.plot(lam[:len(testerr)],testerr,c='royalblue',label=r'${\rm mean\ test\ error}$')
    plt.plot(lam[:len(teststd)],teststd,c='royalblue',linestyle='dashed',label=r'${\rm test\ \sigma }$')
    plt.tick_params(direction='in', top=True, bottom=True, left=True, right='True')
    plt.legend(loc='upper right',frameon=False)
    plt.savefig('QSANNdRA_errors.png',bbox_inches='tight',dpi=300)
    plt.show()


def import_preprocessing_models(path_to_models):
    scaler_one_red = joblib.load(str(path_to_models)+'scaler_one_red.pkl')
    scaler_one_blue = joblib.load(str(path_to_models)+'scaler_one_blue.pkl')
    pca_red = joblib.load(str(path_to_models)+'pca_red.pkl')
    pca_blue = joblib.load(str(path_to_models)+'pca_blue.pkl')
    scaler_two_red = joblib.load(str(path_to_models)+'scaler_two_red.pkl')
    scaler_two_blue = joblib.load(str(path_to_models)+'scaler_two_blue.pkl')

    return scaler_one_red, scaler_one_blue, pca_red, pca_blue, scaler_two_red, scaler_two_blue

def load_reconstructed_file(path_to_file,path_to_norms):
    spec = pd.read_csv(path_to_file)
    spec = spec.drop(spec.columns[0],axis=1)
    loglams = np.array(spec.columns).astype(float)
    red = spec.loc[:,u'3.1107':].values

    norms = pd.read_csv(path_to_norms)
    norms = norms.drop(norms.columns[0],axis=1)
    norm = norms.values[0,0]

    return red, loglams, norm

def apply_QSANNdRA(path_to_files,red,lam_out,norm_lam,path_to_models,specname,norm,C=100):
    
    scaler_one_red, scaler_one_blue, pca_red, pca_blue, scaler_two_red, scaler_two_blue = \
        import_preprocessing_models(path_to_models=str(path_to_models)+'preprocessing_models/')
    
    red_trans = scaler_two_red.transform(pca_red.transform(scaler_one_red.transform(red)))
    
    # Load QSANNdRA and compute blue-side predictions
    blue_dim = len(scaler_one_blue.get_feature_names_out())
    predictions = np.empty((blue_dim,C))
    errs = np.empty((blue_dim,C))
    s = range(1,C+1)
    for i in range (0,C):
        model = get_NN(seed=s[i])
        model = load_NN_weights(model=model,weights=str(path_to_models)+'QSANNdRA/NN_'+str(i)+'.h5')
        predictions[:,i] = scaler_one_blue.inverse_transform(pca_blue.inverse_transform(scaler_two_blue.inverse_transform(model.predict(red_trans))))
        errors = np.loadtxt(str(path_to_models)+'errors/NN_'+str(i)+'_err_std.csv',delimiter=',')
        errs[:,i] = 1-errors[:,0]

    err_norm = errs.sum(axis=1)
    w = np.divide(errs,err_norm[:,None])
    mean_prediction = np.average(predictions,weights=w,axis=1)

    loglam_target_out = np.around(np.arange(np.log10(lam_out[0]),np.log10(lam_out[1]),0.0001),decimals=4)
    norm_loglam = np.around(np.log10(norm_lam),decimals=4)
    loglams = np.delete(loglam_target_out,np.where(loglam_target_out==norm_loglam)[0])
    save_ar = np.concatenate((10**loglams.reshape(-1,1),predictions),axis=1)
    print(np.shape(save_ar))
    np.savetxt(str(path_to_files)+str(specname)+'/'+str(specname)+'_prediction_mean.txt',np.transpose([10**loglams[:len(mean_prediction)],mean_prediction]),delimiter=' ',fmt='%1.10f',\
        header='Mean prediction of QSANNdRA for the spectrum of '+str(specname)+' normalized at 1290 A. Normalization constant: '+str(norm))
    np.savetxt(str(path_to_files)+str(specname)+'/'+str(specname)+'_predictions.txt',\
            save_ar,delimiter=' ',fmt='%1.10f',\
            header='All predictions of QSANNdRA for the spectrum of '+str(specname)+' normalized at 1290 A. Normalization constant: '+str(norm))

    return mean_prediction, predictions


def plot_predictions(path_to_files,path_to_plots,specname,lam_in,lam_out,red,predictions,mean_prediction,norm):
    
    if lam_in[0] == 1290:
        loglam_target_in = np.around(np.arange(np.log10(lam_in[0])+0.0001,np.log10(lam_in[1]),0.0001),decimals=4)
    else:
        loglam_target_in = np.around(np.arange(np.log10(lam_in[0]),np.log10(lam_in[1]),0.0001),decimals=4)
    if lam_out[1] == 1290:
        loglam_target_out = np.around(np.arange(np.log10(lam_out[0]),np.log10(lam_out[1])-0.0001,0.0001),decimals=4)
    else:
        loglam_target_out = np.around(np.arange(np.log10(lam_out[0]),np.log10(lam_out[1]),0.0001),decimals=4)

    spec = fits.open(str(path_to_files)+str(specname)+'.fits')
    z = spec[2].data['Z']
    mask = spec[1].data['and_mask']
    flux = spec[1].data['flux'][mask == 0]
    wave = (10**spec[1].data['loglam'][mask == 0])/(1+z)
    ivar = spec[1].data['ivar'][mask == 0]
    spec.close()

    _, axs = plt.subplots(2,1,sharey=True,figsize=(6.97,3.31))
    axs[0].plot(wave,np.divide(flux,norm),c=(0.25,0.25,0.25),label=r'${\rm ' +str(specname)+'\ raw\ data}$')
    axs[1].plot(wave,np.divide(flux,norm),c=(0.25,0.25,0.25))
    axs[0].plot(wave,np.divide(1/np.sqrt(ivar),norm),c='darkgrey',linewidth=1)#,label=r'${\rm raw\ flux\ errors}$')
    axs[1].plot(wave,np.divide(1/np.sqrt(ivar),norm),c='darkgrey',linewidth=1)
    #NN_err = np.loadtxt('models/errors/QSANNdRA_test_performance.csv',delimiter=',')
    #axs[0].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya)*(1+NN_err[:,0]),c='mediumvioletred',linewidth=2,linestyle='dotted',label=r'${\rm QSANNdRA\ uncertainty:\ }$' r'$\overline{\epsilon}$')
    #axs[1].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya)*(1+NN_err[:,0]),c='mediumvioletred',linewidth=2,linestyle='dotted',label='mean prediction')
    #axs[0].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya)*(1-NN_err[:,0]),c='mediumvioletred',linewidth=2,linestyle='dotted')
    #axs[1].plot(10**loglams[:len(mean_prediction)],np.divide(mean_prediction*norm,norm_Lya)*(1-NN_err[:,0]),c='mediumvioletred',linewidth=2,linestyle='dotted')    
    axs[0].axvline(x=(1215.67),c='k',linestyle='--',linewidth=1)
    axs[1].axvline(x=(1215.67),c='k',linestyle='--',linewidth=1)
    mx = np.max(np.max(predictions))
    axs[0].plot(10**loglam_target_in,red[0,:],c='indianred')#label=r'${\rm flux\ fit}$')
    axs[1].plot(10**loglam_target_in,red[0,:],c='indianred')
    axs[0].plot(10**loglam_target_out,predictions,c='royalblue',linewidth=1,alpha=0.05)
    axs[1].plot(10**loglam_target_out,predictions,c='royalblue',linewidth=1,alpha=0.05)
    axs[0].plot(10**loglam_target_out,mean_prediction,c='royalblue',linewidth=2,label=r'${\rm QSANNdRA: mean\ prediction}$')
    axs[1].plot(10**loglam_target_out,mean_prediction,c='royalblue',linewidth=2)
    axs[0].set_ylabel(r'${\rm normalized\ flux}$')
    axs[0].set_xlim([lam_out[0], lam_in[1]])
    axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
    axs[1].set_ylabel(r'${\rm normalized\ flux}$')
    axs[1].set_xlim([lam_out[0], 1290])
    axs[1].set_ylim([-0.1,1.2*mx])
    axs[0].legend(frameon=False,loc='upper right')
    plt.savefig(str(path_to_plots)+str(specname)+'_prediction.png',bbox_inches='tight',dpi=400)
    plt.show()