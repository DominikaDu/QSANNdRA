# -*- coding: utf-8 -*-
##########################################

# Author: Dominika Ďurovčíková (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# D. Ďurovčíková, H. Katz, S. E. I. Bosman, F. B. Davies, J. Devriendt, and A. Slyz, Monthly Notices of the Royal Astronomical Society 493, 4256 (2020).

##########################################

# This module contains routines for SDSS training data preprocessing. For a full description of the preprocessing
# pipeline, please refer to Ďurovčíková et al. 2019 (https://arxiv.org/abs/1912.01050).

# Preprocessing pipeline: structure:

# 0. Cuts on the QSO catalog
# 1. Download spec files
# 2. SN cut
# 3. mask out sky lines and pixels that were flagged as bad
# 4. recover a smooth curve that captures the main features of each spectrum
# 5. calibrate wavelengths for redshifts
# 6. create a table with objects as rows, log wavelengths as columns and 
#    interpolate a given number of data points for each object
# 7. normalize the flux values for all spectra
# 8. perform a final cut and thus create a training set
# 9. perform a Random Forest based cleaning of the training set

import os
import pyfits as pf
from scipy import interpolate
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RFR
from QSmooth import open_calibrate_fits, mask_SDSS, smooth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import QSANNdRA as Q


def perform_catalog_cut(catalog='DR14Q_v4_4.fits', z_start=2.09, z_end=2.51, out_file=None):
    # Performs a ZWARNING, BAL and redshift cut on the catalog given by catalog.

    hdu = pf.open(str(catalog))
    hdudata = hdu[1].data
    # Leave out quasars with highly uncertain redshifts
    mask_z = hdudata['ZWARNING'] == 0
    hdudata_one = hdudata[mask_z]
    # Exclude BALs
    mask_BAL = hdudata_one['BI_CIV'] == 0.0
    hdudata_two = hdudata_one[mask_BAL]
    # Pick only quasars whose redshift is between z_start and z_end
    mask_range = (hdudata_two['Z_PIPE'] > z_start) & (hdudata_two['Z_PIPE'] < z_end)
    hdudata_three = hdudata_two[mask_range]

    data_out = hdudata_three
    hdu_new = pf.BinTableHDU(data=data_out)
    if out_file != None:
        hdu_new.writeto(str(out_file))
    hdu.close()
    
    return data_out
    
def download_spec_files(hdu,target_directory):
    # Composes a download_SDSS.txt list of quasar spectra to be downloaded from the SAS via
    # wget -i download_SDSS.txt

    hdudata = hdu
    # Create the txt file with writing permissions
    file = open('download_SDSS.txt','w+')
    # Loop over hdudata elements and write the corresponding URL to the file
    for i in range(len(hdudata)):
        if hdudata[i]['SPECTRO'] == 'BOSS':
            file.write('https://data.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/lite/' + str(hdudata[i]['PLATE']).zfill(4) +'/spec-' + \
            str(hdudata[i]['PLATE']).zfill(4) + '-' + str(hdudata[i]['MJD']).zfill(5) + '-' + str(hdudata[i]['FIBERID']).zfill(4) + '.fits\n')
        elif hdudata[i]['SPECTRO'] == 'SDSS':
            file.write('https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/spectra/lite/' + str(hdudata[i]['PLATE']).zfill(4) +'/spec-' + \
            str(hdudata[i]['PLATE']).zfill(4) + '-' + str(hdudata[i]['MJD']).zfill(5) + '-' + str(hdudata[i]['FIBERID']).zfill(4) + '.fits\n')
            file.write('https://data.sdss.org/sas/dr14/sdss/spectro/redux/103/spectra/lite/' + str(hdudata[i]['PLATE']).zfill(4) +'/spec-' + \
            str(hdudata[i]['PLATE']).zfill(4) + '-' + str(hdudata[i]['MJD']).zfill(5) + '-' + str(hdudata[i]['FIBERID']).zfill(4) + '.fits\n')
            file.write('https://data.sdss.org/sas/dr14/sdss/spectro/redux/104/spectra/lite/' + str(hdudata[i]['PLATE']).zfill(4) +'/spec-' + \
            str(hdudata[i]['PLATE']).zfill(4) + '-' + str(hdudata[i]['MJD']).zfill(5) + '-' + str(hdudata[i]['FIBERID']).zfill(4) + '.fits\n')
            #NOTE: this is done as I haven't found a way to get the corresponding RUN2D (26, 103 or 104) from the catalog
            #so two out of three SDSS links will not find a file to download (that's OK)
    file.close()
    cmd = 'wget -i download_SDSS.txt --directory-prefix='+str(target_directory)
    os.system(cmd)

def preprocess_training_data(input_directory, output_spectra_file=None, output_norm_file=None, output_directory_plots='plots/preprocessing/', SN_threshold=7.0, norm_lam = 1290, lam_start=1000, lam_stop=3900):
    # Add description

    norm_loglam = np.around(np.log10(norm_lam),decimals=4)
    loglam_target = np.around(np.arange(np.log10(lam_start),np.log10(lam_stop),0.0001),decimals=4)

    Df = pd.DataFrame(data=[])
    norm = pd.DataFrame(data=[])
    m = 0

    for filename in os.listdir(str(input_directory)):
        if filename.endswith('.fits'):
            print(m)
            print(str(filename))
            spec = pf.open(str(input_directory)+str(filename))
            # Perform SN cut
            if spec[2].data['SN_MEDIAN_ALL'] < SN_threshold:
                spec.close()
                print('closing spectrum')
            else:
                mask = mask_SDSS(filename=filename,path=input_directory)
                if len(spec[1].data['loglam'][mask])<1000:
                    # Too few unmasked data points
                    spec.close()
                    print('closing spectrum')
                else:
                    try:
                        spec.close()
                        # Calibrate wavelengths according to redshift
                        [loglam, flux, err] = open_calibrate_fits(filename=filename,path=input_directory)
                        [loglam_smooth, flux_smooth] = smooth(loglam,flux,err,mask=mask)
                        # Interpolate the smooth spectrum
                        int_spec = interpolate.interp1d(loglam_smooth,flux_smooth,bounds_error=False,fill_value=np.nan)
                        if loglam_smooth[0] >= norm_loglam: # If data points missing on the blue side
                            raise Exception('bad spectrum')
                        elif loglam_smooth[-1] <= norm_loglam: # If data points missing on the red side
                            raise Exception('bad spectrum')
                        else:	
                            int_flux = int_spec(loglam_target) # Interpolate to target lambdas
                            spec_id = [str(filename)]
                            df = pd.DataFrame(data=[int_flux],columns=loglam_target,index=spec_id)
                            norm_flux = df.iloc[0].at[norm_loglam]
                            df_norm = df.divide(norm_flux)
                            Df = Df.append(df_norm)
                            norm_here = pd.DataFrame(data=[norm_flux],index=spec_id)
                            norm = norm.append(norm_here)
                            m += 1
                            if m % 100 == 0:
                                plot_example_spectrum(path=output_directory_plots,loglam=loglam,flux=flux,err=err,loglam_smooth=loglam_target,flux_smooth=int_flux,spec_id=spec_id[0],norm_flux=norm_flux)
                        df = []
                        df_norm = []
                        norm_here = []
                        print('spectrum processed')
                    except:
                        spec.close()
                        print('closing spectrum')

    print('shape after processing:')
    print(Df.shape)
    if output_spectra_file != None:
        Df.to_csv(str(output_spectra_file))
    if output_norm_file != None:
        norm.to_csv(str(output_norm_file))

    return Df, norm

def perform_flux_cut(Df, output_file=None,thres_lam=1280):
    # Performs a set of cuts on the smoothed flux:
    # 1. Reject spectra whose fluxes fall below 0.5 below 128 nm (quasars with strongest associated absorption)
    # 2. Reject spectra whose fluxes fall below 0.1 above 128 nm (remove remaining quasars with poor SN ratio on the red side)
 
    thres_loglam = np.around(np.log10(thres_lam),decimals=4)
    thres_ind = Df.columns.get_loc(thres_loglam)
    Df_new = Df.drop(Df[Df.values[:,1:thres_ind]<0.5].index)
    Df_new = Df_new.drop(Df_new[Df_new.values[:,thres_ind:]<0.1].index)

    print('shape after flux cuts:')
    print(Df_new.shape)
    if output_file != None:
        Df_new.to_csv(str(output_file))

    return Df_new

def prepare_training_set(input_spectra_file,input_norm_file,output_spectra_file,output_norm_file,lam_start=1191.5,lam_stop=2900):
    # Prepares the final training set, to be used for a Random Forest based cleanup.
    # The wavelength range is given by lam_start and lam_stop.

    Df_raw = input_spectra_file
    loglam_start = np.around(np.log10(lam_start),decimals=4)
    loglam_stop = np.around(np.log10(lam_stop),decimals=4)
    start = Df_raw.columns.get_loc(loglam_start)
    stop = Df_raw.columns.get_loc(loglam_stop)
    Df_clean = Df_raw.drop(Df_raw.columns[stop:],axis=1)
    Df_clean = Df_clean.drop(Df_clean.columns[1:start],axis=1)
    # Drop quasars with missing data in this wavelength range.
    Df = Df_clean.dropna(axis=0)
    print(np.shape(Df))
    Df.to_csv(str(output_spectra_file))
    ID_dat = input_norm_file
    ID_df = ID_dat[ID_dat.iloc[:,0].isin(Df.iloc[:,0])]
    ID_df.to_csv(str(output_norm_file))

    return Df, ID_df

def plot_example_spectrum(path,loglam,flux,err,loglam_smooth,flux_smooth,spec_id,norm_flux=None):
    plt.rc('text',usetex=True)
    font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
    plt.rc('font',**font)
    fig, axs = plt.subplots(2,1,sharey=True,figsize=(6.97,3.31))
    if norm_flux == None:
        axs[0].plot(10**loglam,flux,c=(0.25,0.25,0.25),label=r'${\rm ' +str(spec_id)+'\ raw\ data}$')
        axs[1].plot(10**loglam,flux,c=(0.25,0.25,0.25))
        axs[0].plot(10**loglam,1/err,c='r',linewidth=1,label=r'${\rm \ flux\ errors}$')
        axs[1].plot(10**loglam,1/err,c='r',linewidth=1)
        axs[0].plot(10**loglam_smooth,flux_smooth,color='c',label=r'${\rm QSmooth\ fit}$')
        axs[1].plot(10**loglam_smooth,flux_smooth,color='c')
        axs[0].set_ylabel(r'${\rm flux\ [erg\ s}$'+r'$^{-1} {\rm cm} ^{-2}\mathrm{\AA}$'+r'${\rm ]}$')
        axs[1].set_ylabel(r'${\rm flux\ [erg\ s}$'+r'$^{-1} {\rm cm} ^{-2}\mathrm{\AA}$'+r'${\rm ]}$')
        axs[0].set_ylim(bottom=-5)
        axs[1].set_ylim(bottom=-5)
    else:
        axs[0].plot(10**loglam,np.divide(flux,norm_flux),c=(0.25,0.25,0.25),label=r'${\rm ' +str(spec_id)+'\ raw\ data}$')
        axs[1].plot(10**loglam,np.divide(flux,norm_flux),c=(0.25,0.25,0.25))
        axs[0].plot(10**loglam,np.divide(1/err,norm_flux),c='r',linewidth=1,label=r'${\rm \ flux\ errors}$')
        axs[1].plot(10**loglam,np.divide(1/err,norm_flux),c='r',linewidth=1)
        axs[0].plot(10**loglam_smooth,np.divide(flux_smooth,norm_flux),color='c',label=r'${\rm QSmooth\ fit}$')
        axs[1].plot(10**loglam_smooth,np.divide(flux_smooth,norm_flux),color='c')
        axs[0].set_ylabel(r'${\rm normalized\ flux}$')
        axs[1].set_ylabel(r'${\rm normalized\ flux}$')
        axs[0].set_ylim([-0.02,7])
        axs[1].set_ylim([-0.02,7])
    axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
    axs[0].set_xlim([1150, 2900])
    axs[1].set_xlim([1150, 1500])
    axs[0].legend(frameon=False)
    plt.savefig(str(path)+str(spec_id)+'_example.png',bbox_inches='tight',dpi=300)
    plt.close()

def RF_cleanup(spectra_file,norms_file,path_to_clean_spectra,path_to_bad_spectra,path_to_clean_norms,path_to_bad_norms,RF_num=100,pca_variance=0.99,ratio=0.8,norm_lam=1290,n_folds=10):
    spectra = pd.read_csv(str(spectra_file))
    norms = pd.read_csv(str(norms_file))

    norm_loglam = np.around(np.log10(norm_lam),decimals=4)

    spectra = spectra.drop(columns=str(norm_loglam)) # Drop 1290 column, as all spectra have flux normalized to 1 here (no variance)
    spectra = spectra.drop(spectra.columns[0:2],axis=1) # These contain row numbers and spectrum ids

    norms = norms.drop(norms.columns[0],axis=1) # This contains row numbers
    loglams = np.array(spectra.columns).astype(np.float) # Extract wavelengths

    # Divide spectra into input and output arrays
    blue = spectra.loc[:,:str(norm_loglam)].values
    red = spectra.loc[:,str(norm_loglam+0.0001):].values

    print(np.shape(blue))
    print(np.shape(red))
    print(np.shape(loglams))
    # K-fold split for cleanup
    skfolds = KFold(n_splits=n_folds, random_state=0, shuffle=False)
    i = 1 # Counter
    spectra_nan = pd.DataFrame(data=[])
    for train_id, test_id in skfolds.split(red,blue):
        print('fold '+str(i))
        train_red_orig = red[train_id]
        train_blue_orig = blue[train_id]
        test_red_orig = red[test_id]
        test_blue_orig = blue[test_id]

        _, train_red, test_red = Q.standardize(train_red_orig,test_red_orig)
        scaler_blue, train_blue, test_blue = Q.standardize(train_blue_orig,test_blue_orig)
        _, train_red, test_red = Q.perform_PCA(train_red,test_red,n_components=pca_variance)
        pca_blue, train_blue, test_blue = Q.perform_PCA(train_blue,test_blue,n_components=pca_variance)

        reg = RFR(n_estimators=RF_num,random_state=0)
        reg.fit(train_red,train_blue)
        rf_blue = reg.predict(test_red)

        rf_blue_trans = scaler_blue.inverse_transform(pca_blue.inverse_transform(rf_blue))

		# Reject data points in the test fold
        err = ((np.abs(rf_blue_trans - test_blue_orig)/test_blue_orig)).mean(axis=0) # Calculate mean absolute error for each wavelength
        std = ((np.abs(rf_blue_trans - test_blue_orig)/test_blue_orig)).std(axis=0) # Calculate std of absolute error for each wavelength
		# Reject data points that lie above 3 standard deviations away from mean
        test_blue_clean = np.where(((rf_blue_trans - test_blue_orig)/test_blue_orig) < err+3*std,test_blue_orig,np.nan)
        rf_blue_clean = np.where(((rf_blue_trans - test_blue_orig)/test_blue_orig) < err+3*std,rf_blue_trans,np.nan)

        # Calculate fraction of rejected data points
        rejected = sum(sum(np.isnan(rf_blue_clean)))
        total = np.int(np.size(rf_blue_clean))
        print('rejected '+np.str(np.float(rejected)/np.float(total)))
        print(np.shape(test_blue_clean))
        print(np.shape(test_red_orig))
        # Append the clean arrays to prepared dataframes
        test_new = np.concatenate((test_blue_clean,test_red_orig),axis=1)
        test_df = pd.DataFrame(data=test_new,columns=loglams,index=test_id)
        spectra_nan = spectra_nan.append(test_df)

        i += 1
    
    # Drop spectra with rejected data points
    spectra_clean = spectra_nan.dropna(axis=0,how='any')
    spectra_clean.to_csv(str(path_to_clean_spectra))
    norms_clean = norms[norms.iloc[:,1].isin(spectra_clean.iloc[:,0])]
    norms_clean = norms_clean.drop(norms.columns[0],axis=1)
    norms_clean.to_csv(str(path_to_clean_norms))
	
    spectra_bad = spectra_nan[~spectra_nan.index.isin(spectra_clean.index)]
    spectra_bad_full = spectra.iloc[spectra_bad.index]
    spectra_bad_full.to_csv(str(path_to_bad_spectra))
    norms_bad = norms[norms.iloc[:,1].isin(spectra_bad.iloc[:,0])]
    norms_bad = norms_bad.drop(norms.columns[0],axis=1)
    norms_bad.to_csv(str(path_to_bad_norms))

    return spectra_clean, norms_clean, spectra_bad, norms_bad




