# -*- coding: utf-8 -*-
##########################################

# Author: Dominika Ďurovčíková (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# Ďurovčíková, D., Katz, H., Bosman, S.E.I., Davies, F.B., Devriendt, J. and Slyz, A., 2019.
# Reionization history constraints from neural network based predictions of high-redshift quasar continua.
# arXiv preprint arXiv:1912.01050.

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

import os
import pyfits as pf
from scipy import interpolate
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn import linear_model
from QSmooth import open_calibrate_fits, mask_SDSS, smooth
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def perform_catalog_cut(catalog='DR14Q_v4_4.fits', z_start=2.09, z_end=2.51, out_file='QSO_catalog_cut.fits'):
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

    hdu_new = pf.BinTableHDU(data=hdudata_three[:100])
    hdu_new.writeto(str(out_file))
    hdu.close()
    return hdu_new
    
def download_spec_files(filename,target_directory):
    # Composes a download_SDSS.txt list of quasar spectra to be downloaded from the SAS via
    # wget -i download_SDSS.txt

    hdu = pf.open(filename)
    hdudata = hdu[1].data
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
    hdu.close()
    # !!Downloading to a target directory is not working!!
    cmd = 'wget -i download_SDSS.txt --output '+str(target_directory)
    os.system(cmd)

def preprocess_training_data(input_directory, output_directory, output_directory_plots='plots/', SN_threshold=7.0, norm_lam = 1290, lam_start=1000, lam_stop=3900):
    # Add description

    norm_loglam = np.around(np.log10(norm_lam),decimals=4)
    loglam_target = np.around(np.arange(np.log10(lam_start),np.log10(lam_stop),0.0001),decimals=4)

    Df = pd.DataFrame(data=[])
    norm = pd.DataFrame(data=[])
    m = -1

    for filename in os.listdir(str(input_directory)):
        if filename.endswith('.fits'):
            m += 1
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
                            if m % 1 == 0:
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
    Df.to_csv(str(output_directory)+str(SN_threshold)+'SN_QSO_spectra_primary.csv')
    norm.to_csv(str(output_directory)+str(SN_threshold)+'SN_QSO_spectra_norm_primary.csv')

    return Df, norm

def perform_flux_cut(Df, output_directory, SN_threshold=7.0, thres_lam=1280):
    # Performs a set of cuts on the smoothed flux:
    # 1. Reject spectra whose fluxes fall below 0.5 below 128 nm (quasars with strongest associated absorption)
    # 2. Reject spectra whose fluxes fall below 0.1 above 128 nm (remove remaining quasars with poor SN ratio on the red side)
 
    thres_loglam = np.around(np.log10(thres_lam),decimals=4)
    thres_ind = Df.columns.get_loc(thres_loglam)
    Df_new = Df.drop(Df[Df.values[:,1:thres_ind]<0.5].index)
    Df_new = Df_new.drop(Df_new[Df_new.values[:,thres_ind:]<0.1].index)

    print('shape after flux cuts:')
    print(Df_new.shape)
    Df_new.to_csv(str(output_directory)+str(SN_threshold)+'SN_QSO_spectra_preprocessed.csv')

    return Df_new

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
        axs[0].set_ylim([-0.02,5])
        axs[1].set_ylim([-0.02,5])
    axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
    axs[0].set_xlim([1150, 2900])
    axs[1].set_xlim([1150, 1500])
    axs[0].legend(frameon=False)
    plt.savefig(str(path)+str(spec_id)+'_example.png',bbox_inches='tight',dpi=300)
    plt.close()