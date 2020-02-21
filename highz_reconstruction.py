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

    [loglam_smooth, flux_smooth] = QSmooth.smooth(loglam,flux,f_err,mask=[],bin_s=10,shuf=5,Lya=False)
    int_spec = interpolate.interp1d(loglam_smooth,flux_smooth)
    norm_df = np.float(int_spec(norm_loglam))
    flux_int = np.divide(int_spec(loglam_target),norm_df)

    df_norm = pd.DataFrame(data=[flux_int],columns=np.asarray(loglam_target,dtype='str'))
    df_norm = df_norm.drop(columns=str(norm_loglam))

    return df_norm