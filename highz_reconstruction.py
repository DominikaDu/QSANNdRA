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
reload(QSmooth)
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
        _, model = Q.train_NN(model=model,train_red=train_x,train_blue=train_y,checkpointer_path=str(path_to_NN),epochs=400,batch_size=800)

    return model 

def prepare_high_z_data(raw_data,z_quasar,skiprows,na_values,norm_lam=1290,lam_start=1191.5,lam_stop=2900,bin_size=10,shuffle=5):
    
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

    #mask = np.zeros(len(loglam), dtype=np.int)
    #for i in range(0,len(loglam)):
    #    if np.isnan(f_err[i]):
    #        mask[i] = 1
    #    if flux[i]< 3e-18:
    #        mask[i] = 1
    #mask_bool = mask==0
    wave = pd.Series(data.values[:,0])
    mask = get_masked_values(wave)

    [loglam_smooth, flux_smooth] = QSmooth.smooth(loglam,flux,1-(f_err)**4,mask= ~mask,bin_s=bin_size,shuf=shuffle,Lya=False)
    int_spec = interpolate.interp1d(loglam_smooth,flux_smooth,fill_value='extrapolate')
    norm = np.float(int_spec(norm_loglam))
    norm_Lya = np.float(int_spec(Lya_loglam))
    flux_int = np.divide(int_spec(loglam_target),norm)

    df_norm = pd.DataFrame(data=[flux_int],columns=np.asarray(loglam_target,dtype='str'))
    df_norm = df_norm.drop(columns=str(norm_loglam))

    return df_norm, norm, norm_Lya, loglam, flux, f_err, mask

def plot_smoothed_spectrum(loglam_raw,flux_raw,f_err_raw,loglams,spec,norm,name_plot,name_file,mask=None):
    plt.rc('text',usetex=True)
    font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
    plt.rc('font',**font)
    plt.figure(figsize=(6.97,1.60))
    if len(mask)>0:
        plt.vlines(x=(10**loglam_raw[mask]),ymin=-0.1,ymax=3.5,color='lightgray')
    plt.plot(10**loglam_raw,np.divide(flux_raw,norm),c=(0.25,0.25,0.25),label=str(name_plot)+r'${\rm \ raw\ data}$')
    plt.plot(10**loglams,spec.values[0,:],c='c',label=r'${\rm QSmooth\ fit}$')
    plt.plot(10**loglam_raw,np.divide(f_err_raw,norm),c='r',linewidth=0.5,label=r'${\rm raw\ flux\ errors}$')
    plt.ylim([-0.1,3])
    plt.xlim([1180,2910])
    plt.legend(frameon=False)
    plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
    plt.ylabel(r'${\rm normalized\ flux}$')
    plt.savefig('plots/high-z/'+str(name_file)+'_smooth.png',bbox_inches='tight',dpi=400)



# From Eduardo


def get_masked_values(wave):
    """
    The wavelength windows I’m masking are based on the full resolution spectrum, and correspond to the following velocity/wavelength ranges:

    """
    #DLA
    m1 = wave.between(9326., 9339) #SiII 1260 #Overlap with CIV z=5.0172 absorber
    m2 = wave.between(9630., 9645.)#including OI 1302
    m17 = wave.between(9655., 9660.)# Si II 1304
    m3 = wave.between(9875., 9886.)# CII 1334
    m19 = wave.between(11298., 11311.)# Si II 1526
    m20 = wave.between(12366., 12374.)# Al II 1670
    m21 = wave.between(17631., 17667.)# Fe II 2382 + sky
    m22 = wave.between(20692., 20723.)# MgII 2796
    m18 = wave.between(20748., 20770.)# MgII 2803
    
    #z=6.0645
    m4 = wave.between(9196, 9202.) #OI 1302
    m5 = wave.between(9213., 9217.) #Si II 1304
    m6 = wave.between(9423., 9433.) # CI 1334
    m7 = wave.between(9840., 9854.) #SiIV 1393
    m23 = wave.between(10780., 10788.) #Si II 1526
    m24 = wave.between(10931., 10946.) #CIV 1548
    m25 = wave.between(10949., 10964.)#CIV 1550
    m26 = wave.between(19747., 19763.)#MgII 2796
    m27 = wave.between(19796., 19812.)#MgII 2803
    
    #z=5.8434  absorber
    m10 = wave.between(9124., 9140.) #CII 1334
    m28 = wave.between(16299, 16315) #Fe II 2382
    m29 = wave.between(19514, 19538) #Mg I 2852
    
    #z=5.0172 CIV absorber
    m8 = wave.between(9311, 9322) #CIV 1548
    m9 = wave.between(9327, 9338) #CIV 1550
    
    #z=3.4185 absorber
    m30 = wave.between(12349, 10887) ##FeII 2586
    m31 = wave.between(10923, 10947) ##FeII 2600
    m32 = wave.between(11757, 11770) #MgII 2796
    m33 = wave.between(11789, 11800) #MgI 2852
    
    #UNKNOWN absorbers
   
    m11 = wave.between(9091., 9100.)
    m12 = wave.between(9075., 9081.)
    #SKY emission lines
    m13 = wave.between(8989., 8992.5)
    m14 = wave.between(8999.5, 9002.)
    m15 = wave.between(9572.4, 9577.6)
    #m16 = between(wave, 9100.1, 9105.5)
    m16 = wave.between(12822, 12829)
    m = m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | m15 | m16  | m17
    m |= m18 | m19 | m20 | m21 | m22 | m23 | m24 | m25 | m26 | m27 | m28 | m29
    m |= m30 | m31 | m32 | m33
    
    return m