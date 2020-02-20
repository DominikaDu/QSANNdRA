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

def save_model(model,target_path):
    joblib.dump(model,str(path))

def load_training_data(path_to_spectra,path_to_norms):
    spectra = pd.read_csv(path_to_spectra)
    norms = pd.read_csc(path_to_norms)
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

    loglam = np.around(np.log10(lam),decimals=4)+0.0001
    train_blue = train.loc[:,:str(loglam)].values
    train_red = train.loc[:,str(loglam):].values
    test_blue = test.loc[:,:str(loglam)].values
    test_red = test.loc[:,str(loglam):].values

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

def transform_training_data(train_orig,test_orig,n_comp,save=False,target_path=None):
    scaler_one, train, test = standardize(train=train_orig,test=test_orig,save=save,target_path=target_path)
    pca, train, test = perform_PCA(train=train_orig,test=test_orig,n_components=n_comp,save=save,target_path=target_path)
    scaler_two, train, test = standardize(train=train_orig,test=test_orig,save=save,target_path=target_path)

    return scaler_one, pca, scaler_two, train, test

def prepare_training_data(training_spectra,training_norms,models_save=False,path_to_models=None):
    # Loads and splits training data into train and test sets and performs a sequence of standardization, PCA, standardization.

    spectra, norms = Q.load_training_data(path_to_spectra=training_spectra,path_to_norms=training_norms)
    lam = Q.extract_lambdas(dataframe=spectra)
    train_red_orig, train_blue_orig, test_red_orig, test_blue_orig = \
        train_test_split(dataframe=spectra,ratio=0.8,lam=1290)
    scaler_red_one, pca_red, scaler_red_two, train_red, test_red = \
        transform_training_data(train_orig=train_red_orig,test_orig=test_red_orig,n_comp=63,save=models_save,target_path=path_to_models)
    scaler_blue_one, pca_blue, scaler_blue_two, train_blue, test_blue = \
        transform_training_data(train_orig=train_blue_orig,test_orig=test_blue_orig,n_comp=36,save=models_save,target_path=path_to_models)

    return scaler_red_one, pca_red, scaler_red_two, train_red, test_red, scaler_blue_one, pca_blue, scaler_blue_two, train_blue, test_blue

def get_QSANNdRA(red_comp=63,n_two=40,n_three=40,blue_comp=36,seed=0,act='elu',optimizer='adam'):
	model = Sequential()
	model.add(Dense(n_two, input_dim=red_comp, kernel_initializer=initializers.RandomNormal(seed=seed),bias_initializer=initializers.RandomNormal(seed=seed),activation=act))
	model.add(Dense(n_three, kernel_initializer=initializers.RandomNormal(seed=seed),bias_initializer=initializers.RandomNormal(seed=seed),activation=act))
	model.add(Dense(blue_comp, kernel_initializer=initializers.RandomNormal(seed=seed),bias_initializer=initializers.RandomNormal(seed=seed)))
	model.compile(loss='mean_absolute_error',optimizer=optimizer,metrics=['mae'])
    #return model

def load_QSANNdRA_weights(model,weights):
    model.load_weights(weights)

    return model

def train_QSANNdRA(model,train_red,train_blue,checkpointer_path,epochs=80,batch_size=500,validation_split=0.2):

	checkpointer = ModelCheckpoint(filepath=checkpointer_path,verbose=1,save_best_only=True,monitor='val_loss',mode='min',period=5)
	history = model.fit(train_red, train_blue, epochs=epochs, batch_size=batch_size,verbose=1, shuffle=True,validation_split=validation_split,callbacks=[checkpointer])
	
    #return history, model 


