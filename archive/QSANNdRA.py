# -*- coding: utf-8 -*-
##########################################

# Author: Dominika Ďurovčíková (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# D. Ďurovčíková, H. Katz, S. E. I. Bosman, F. B. Davies, J. Devriendt, and A. Slyz, Monthly Notices of the Royal Astronomical Society 493, 4256 (2020).


##########################################

# This module contains routines to build and train QSANNdRA. For a full description of the
# procedure, please refer to Ďurovčíková et al. 2019 (https://arxiv.org/abs/1912.01050).


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyfits as pf
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import initializers
import preprocessing as pr

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
    s = range(1,C+1)
    for i in range (0,C):
        print(i)
        model = get_NN(seed=s[i])
        if load:
            model = load_NN_weights(model=model,weights=str(path_to_models)+'QSANNdRA/NN_committee_'+str(i)+'.h5')
        else:
            history, model = train_NN(model=model,train_red=train_red,train_blue=train_blue,checkpointer_path=str(path_to_models)+'QSANNdRA/NN_'+str(i)+'.h5')
        
        # Inverse transform and evaluate errors
        pred_test = scaler_one_blue.inverse_transform(pca_blue.inverse_transform(scaler_two_blue.inverse_transform(model.predict(test_red))))
        predictions_test[:,:,i] = pred_test

        err = ((np.abs(pred_test - test_blue_orig)/test_blue_orig)).mean(axis=0) # Calculate mean absolute relative error for each wavelength
        std = ((np.abs(pred_test - test_blue_orig)/test_blue_orig)).std(axis=0)
        if save_errs:
            np.savetxt(str(path_to_models)+'errors/NN_'+str(i)+'_err_std.csv',np.transpose([err,std]),delimiter=',')
        
        errs[:,i] = 1-err

    # Calculate weighted mean of predictions
    err_norm = errs.sum(axis=1)
    w = np.divide(errs,err_norm[:,None])
    w_test = np.repeat(w[np.newaxis,:, :], len(predictions_test), axis=0)
    mean_preds_test = np.average(predictions_test,weights=w_test,axis=2)

    err_mean = ((np.abs(mean_preds_test - test_blue_orig)/test_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength
    std_mean = ((np.abs(mean_preds_test - test_blue_orig)/test_blue_orig)).std(axis=0) #calculate std squared relative error for each wavelength
    np.savetxt(str(path_to_models)+'errors/QSANNdRA_test_performance.csv',np.transpose([err_mean,std_mean]),delimiter=',')
    print(np.mean(err_mean))
    print(np.mean(std_mean))

    return predictions_test, mean_preds_test



