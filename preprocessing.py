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
# 9. perform a Random Forest based cleaning of the training set

import os
from astropy.io import fits
from astropy import constants as c
from scipy import interpolate
import pandas as pd
import numpy as np
from numpy.random import randint
from scipy.signal import find_peaks
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from QSmooth import spline_smooth, ivarsmooth, view_smooth_spectrum
from csaps import csaps
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr
plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
import QSANNdRA as Q

def perform_catalog_cut(catalog, z_start=2.09, z_end=2.51, snr=5.0, out_file=None, uncertain_BALs=False):
    # Performs a ZWARNING, BAL and redshift cut on the catalog given by catalog.

    hdu = fits.open(str(catalog))
    hdudata = hdu[1].data
    print(np.shape(hdudata))
    # Leave out quasars with highly uncertain redshifts
    mask_z = hdudata['ZWARNING'] == 0
    hdudata_one = hdudata[mask_z]
    print(np.shape(hdudata_one))
    # Exclude BALs
    if uncertain_BALs == True:
        mask_BAL = np.any([hdudata_one['BI_CIV'] == 0.0,hdudata_one['BI_CIV'] == -1.0],axis=0)
    elif uncertain_BALs == False:
        mask_BAL = hdudata_one['BAL_PROB'] == 0
    hdudata_two = hdudata_one[mask_BAL]
    print(np.shape(hdudata_two))
    # Exclude DLAs
    mask_DLA = np.sum(hdudata_two['Z_DLA'],axis=1) == -5
    hdudata_three = hdudata_two[mask_DLA]
    print(np.shape(hdudata_three))
    # SNR cut
    mask_snr = hdudata_three['SN_MEDIAN_ALL'] >= snr
    hdudata_four = hdudata_three[mask_snr]
    print(np.shape(hdudata_four))
    # Pick only quasars whose redshift is between z_start and z_end
    mask_range = (hdudata_four['Z_PIPE'] > z_start) & (hdudata_four['Z_PIPE'] < z_end)
    hdudata_five = hdudata_four[mask_range]

    data_out = hdudata_five
    print(np.shape(data_out))
    hdu_new = fits.BinTableHDU(data=data_out)
    if out_file != None:
        hdu_new.writeto(str(out_file))
    hdu.close()
    
    return data_out

def download_spec_files(hdudata,target_directory,tag):
    # Composes a download_SDSS.txt list of quasar spectra to be downloaded from the SAS via
    # wget -i download_SDSS.txt

    # Create the txt file with writing permissions
    file = open('download_SDSS'+str(tag)+'.txt','w+')
    # Loop over hdudata elements and write the corresponding URL to the file
    for i in range(len(hdudata)):
        file.write('https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/full/' + str(hdudata[i]['PLATE']).zfill(4) +'/spec-' + \
            str(hdudata[i]['PLATE']).zfill(4) + '-' + str(hdudata[i]['MJD']).zfill(5) + '-' + str(hdudata[i]['FIBERID']).zfill(4) + '.fits\n')
        file.write('https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/' + str(hdudata[i]['PLATE']).zfill(4) +'/spec-' + \
            str(hdudata[i]['PLATE']).zfill(4) + '-' + str(hdudata[i]['MJD']).zfill(5) + '-' + str(hdudata[i]['FIBERID']).zfill(4) + '.fits\n')
        file.write('https://data.sdss.org/sas/dr16/sdss/spectro/redux/103/spectra/' + str(hdudata[i]['PLATE']).zfill(4) +'/spec-' + \
            str(hdudata[i]['PLATE']).zfill(4) + '-' + str(hdudata[i]['MJD']).zfill(5) + '-' + str(hdudata[i]['FIBERID']).zfill(4) + '.fits\n')
        file.write('https://data.sdss.org/sas/dr16/sdss/spectro/redux/104/spectra/' + str(hdudata[i]['PLATE']).zfill(4) +'/spec-' + \
            str(hdudata[i]['PLATE']).zfill(4) + '-' + str(hdudata[i]['MJD']).zfill(5) + '-' + str(hdudata[i]['FIBERID']).zfill(4) + '.fits\n')
            #NOTE: this is done as I haven't found a way to get the corresponding RUN2D (26, 103 or 104) from the catalog
            #so two out of three SDSS links will not find a file to download (that's OK)
    file.close()

def compute_stack(wave_grid, waves, fluxes, ivars, masks, weights):
    '''
        Compute a stacked spectrum from a set of exposures on the specified wave_grid with proper treatment of
        weights and masking. This code uses np.histogram to combine the data using NGP and does not perform any
        interpolations and thus does not correlate errors. It uses wave_grid to determine the set of wavelength bins that
        the data are averaged on. The final spectrum will be on an ouptut wavelength grid which is not the same as wave_grid.
        The ouput wavelength grid is the weighted average of the individual wavelengths used for each exposure that fell into
        a given wavelength bin in the input wave_grid. This 1d coadding routine thus maintains the independence of the
        errors for each pixel in the combined spectrum and computes the weighted averaged wavelengths of each pixel
        in an analogous way to the 2d extraction procedure which also never interpolates to avoid correlating erorrs.
        
        Args:
        wave_grid: ndarray, (ngrid +1,)
        new wavelength grid desired. This will typically be a reguarly spaced grid created by the get_wave_grid routine.
        The reason for the ngrid+1 is that this is the general way to specify a set of  bins if you desire ngrid
        bin centers, i.e. the output stacked spectra have ngrid elements.  The spacing of this grid can be regular in
        lambda (better for multislit) or log lambda (better for echelle). This new wavelength grid should be designed
        with the sampling of the data in mind. For example, the code will work fine if you choose the sampling to be
        too fine, but then the number of exposures contributing to any given wavelength bin will be one or zero in the
        limiting case of very small wavelength bins. For larger wavelength bins, the number of exposures contributing
        to a given bin will be larger.
        waves: ndarray, (nspec, nexp)
        wavelength arrays for spectra to be stacked. Note that the wavelength grids can in general be different for
        each exposure and irregularly spaced.
        fluxes: ndarray, (nspec, nexp)
        fluxes for each exposure on the waves grid
        ivars: ndarray, (nspec, nexp)
        Inverse variances for each exposure on the waves grid
        masks: ndarray, bool, (nspec, nexp)
        Masks for each exposure on the waves grid. True=Good.
        weights: ndarray, (nspec, nexp)
        Weights to be used for combining your spectra. These are computed using sn_weights
        Returns:
        wave_stack, flux_stack, ivar_stack, mask_stack, nused
        
        wave_stack: ndarray, (ngrid,)
        Wavelength grid for stacked spectrum. As discussed above, this is the weighted average of the wavelengths
        of each spectrum that contriuted to a bin in the input wave_grid wavelength grid. It thus has ngrid
        elements, whereas wave_grid has ngrid+1 elements to specify the ngrid total number of bins. Note that
        wave_stack is NOT simply the wave_grid bin centers, since it computes the weighted average.
        flux_stack: ndarray, (ngrid,)
        Final stacked spectrum on wave_stack wavelength grid
        ivar_stack: ndarray, (ngrid,)
        Inverse variance spectrum on wave_stack wavelength grid. Erors are propagated according to weighting and
        masking.
        mask_stack: ndarray, bool, (ngrid,)
        Mask for stacked spectrum on wave_stack wavelength grid. True=Good.
        nused: ndarray, (ngrid,)
        Numer of exposures which contributed to each pixel in the wave_stack. Note that this is in general
        different from nexp because of masking, but also becuse of the sampling specified by wave_grid. In other
        words, sometimes more spectral pixels in the irregularly gridded input wavelength array waves will land in
        one bin versus another depending on the sampling.
        '''
    
    ubermask = masks & (weights > 0.0) & (waves > 1.0) & (ivars > 0.0)
    waves_flat = waves[ubermask].flatten()
    fluxes_flat = fluxes[ubermask].flatten()
    ivars_flat = ivars[ubermask].flatten()
    vars_flat = 1 / ivars_flat
    weights_flat = weights[ubermask].flatten()
    
    # Counts how many pixels in each wavelength bin
    nused, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False)
    
    # Calculate the summed weights for the denominator
    weights_total, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False,weights=weights_flat)
    
    # Calculate the stacked wavelength
    wave_stack_total, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False,weights=waves_flat*weights_flat)
    wave_stack = (weights_total > 0.0)*wave_stack_total/(weights_total+(weights_total==0.))
    
    # Calculate the stacked flux
    flux_stack_total, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False,weights=fluxes_flat*weights_flat)
    flux_stack = (weights_total > 0.0)*flux_stack_total/(weights_total+(weights_total==0.))
    
    # Calculate the stacked ivar
    var_stack_total, wave_edges = np.histogram(waves_flat,bins=wave_grid,density=False,weights=vars_flat*weights_flat**2)
    var_stack = (weights_total > 0.0)*var_stack_total/(weights_total+(weights_total==0.))**2
    ivar_stack = 1 / var_stack
    
    # New mask for the stack
    mask_stack = (weights_total > 0.0) & (nused > 0.0)
    
    return wave_stack, flux_stack, ivar_stack, mask_stack, nused


def perform_catalog_cut(catalog, z_start=2.09, z_end=2.51, snr=5.0, out_file=None, uncertain_BALs=False):
    # Performs a ZWARNING, BAL and redshift cut on the catalog given by catalog.

    hdu = fits.open(str(catalog))
    hdudata = hdu[1].data
    print(np.shape(hdudata))
    # Leave out quasars with highly uncertain redshifts
    mask_z = hdudata['ZWARNING'] == 0
    hdudata_one = hdudata[mask_z]
    print(np.shape(hdudata_one))
    # Exclude BALs
    if uncertain_BALs == True:
        mask_BAL = np.any([hdudata_one['BI_CIV'] == 0.0,hdudata_one['BI_CIV'] == -1.0],axis=0)
    elif uncertain_BALs == False:
        mask_BAL = hdudata_one['BAL_PROB'] == 0
    hdudata_two = hdudata_one[mask_BAL]
    print(np.shape(hdudata_two))
    # Exclude DLAs
    mask_DLA = np.sum(hdudata_two['Z_DLA'],axis=1) == -5
    hdudata_three = hdudata_two[mask_DLA]
    print(np.shape(hdudata_three))
    # SNR cut
    mask_snr = hdudata_three['SN_MEDIAN_ALL'] >= snr
    hdudata_four = hdudata_three[mask_snr]
    print(np.shape(hdudata_four))
    # Pick only quasars whose redshift is between z_start and z_end
    mask_range = (hdudata_four['Z_PIPE'] > z_start) & (hdudata_four['Z_PIPE'] < z_end)
    hdudata_five = hdudata_four[mask_range]

    data_out = hdudata_five
    print(np.shape(data_out))
    hdu_new = fits.BinTableHDU(data=data_out)
    if out_file != None:
        hdu_new.writeto(str(out_file))
    hdu.close()
    
    return data_out

def mask_SDSS(filename,path=None):
    # Opens an SDSS spec-PLATE-MJD-FIBER.fits file and creates a mask rejecting SDSS sky lines (Table 30 of Stoughton et al. 2002
    # https://ui.adsabs.harvard.edu/abs/2002AJ....123..485S/abstract) and data points flagged by SDSS pipelines.

    spec = fits.open(str(path)+str(filename))
    data = spec[1].data
    mask = np.zeros(len(data['loglam']), dtype=np.int)
    for i in range(1,len(data['loglam'])):
        if data['and_mask'][i] != 0:
            mask[i] = 1	
            # If in vicinity of SDSS sky lines, then mask out. Can adjust absolute tolerance as required.
            if np.isclose(data['loglam'][i],3.7465,atol=0.002) == True:
                mask[i] = 1
            if np.isclose(data['loglam'][i],3.7705,atol=0.002) == True:
                mask[i] = 1
            if np.isclose(data['loglam'][i],3.7995,atol=0.002) == True:
                mask[i] = 1
            if np.isclose(data['loglam'][i],3.8601,atol=0.002) == True:
                mask[i] = 1
    mask_bool = mask==0
    spec.close()
    return mask_bool

def open_calibrate_fits_old(filename,path):
    hdu_raw = fits.open(str(path)+str(filename))
    # Either calibrate to rest wavelengths based on redshift from a particular line, e.g. Mg-II
    # https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/PLATE4/RUN1D/spZline.html
	# loglam = hdu['loglam'] - np.log10(1+hdu_raw[3].data['LINEZ'][5])
    # Or calibrate to rest wavelengths based on the redshift from the SDSS pipelines.
    loglam = hdu_raw[1].data['loglam'] - np.log10(1+hdu_raw[2].data['Z'])
    flux = hdu_raw[1].data['flux']
    err = hdu_raw[1].data['ivar']
    hdu_raw.close()
    mask = mask_SDSS(filename=filename,path=path)
    return loglam[mask], flux[mask], err[mask]

def open_calibrate_fits(filename,path):
    hdu_raw = fits.open(str(path)+str(filename))
    # Either calibrate to rest wavelengths based on redshift from a particular line, e.g. Mg-II
    # https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/PLATE4/RUN1D/spZline.html
	# loglam = hdu['loglam'] - np.log10(1+hdu_raw[3].data['LINEZ'][5])
    # Or calibrate to rest wavelengths based on the redshift from the SDSS pipelines.
    loglam = hdu_raw[1].data['loglam'] - np.log10(1+hdu_raw[2].data['Z'])
    flux = hdu_raw[1].data['flux']
    err = hdu_raw[1].data['ivar']
    hdu_raw.close()
    return loglam, flux, err

def preprocess_training_data(qsos,input_directory,output_spectra_file=None, output_norm_file=None, output_directory_plots='plots/preprocessing/', SN_threshold=7.0, norm_lam = 1290, lam_in=[1170,1290], lam_out=[1290,2900], version='0'):

    norm_loglam = np.around(np.log10(norm_lam),decimals=4)
    loglam_target = np.around(np.arange(np.log10(lam_out[0]),np.log10(lam_in[1]),0.0001),decimals=4)

    Df = pd.DataFrame(data=[])
    norm = pd.DataFrame(data=[])
    m = 0

    file = open('SDSS_preprocessing_log_version_'+str(version)+'.txt','w+')
    for i in range(0,len(qsos)):
        filename = 'spec-' + str(qsos[i]['PLATE']).zfill(4) + '-' + str(qsos[i]['MJD']).zfill(5) + '-' + str(qsos[i]['FIBERID']).zfill(4) + '.fits'
        file.write(str(m)+'\n')
        print(m)
        file.write(str(filename)+'\n')
        file.flush()
        try:
            spec = fits.open(str(input_directory)+str(filename))
            # Perform SN cut
            if spec[2].data['SN_MEDIAN_ALL'] < SN_threshold:
                file.write(str(spec[2].data['SN_MEDIAN_ALL'])+'\n')
                spec.close()
                file.write('low SNR, closing spectrum\n')
                file.flush()
            else:
                mask = mask_SDSS(filename=filename,path=input_directory)
                if len(spec[1].data['loglam'][mask])<1000:
                    # Too few unmasked data points
                    spec.close()
                    file.write('too few unmasked data points, closing spectrum\n')
                    file.flush()
                else:
                    try:
                        file.write(str(spec[2].data['SN_MEDIAN_ALL'])+'\n')
                        # Calibrate wavelengths according to redshift
                        [loglam, flux, err] = open_calibrate_fits(filename=filename,path=input_directory)
                        if spec[2].data['SN_MEDIAN_ALL'] > 8:
                            int_spec, outlier_mask = spline_smooth(loglam,flux,1/np.sqrt(err),smooth_b=0.999999999,smooth_r=0.9999999,thr=2.0)
                        else:
                            int_spec, outlier_mask = spline_smooth(loglam,flux,1/np.sqrt(err),smooth_b=0.99999999,smooth_r=0.999999,thr=2.0)
                        snr = spec[2].data['SN_MEDIAN_ALL'][0]
                        spec.close()
                        if loglam[0] >= norm_loglam: # If data points missing on the blue side
                            raise Exception('bad spectrum')
                        elif loglam[-1] <= norm_loglam: # If data points missing on the red side
                            raise Exception('bad spectrum')
                        else:	
                            int_flux = int_spec(loglam_target) # Interpolate to target lambdas
                            spec_id = [str(filename)]
                            df = pd.DataFrame(data=[int_flux],columns=loglam_target,index=spec_id)
                            norm_flux = df.iloc[0].at[norm_loglam]
                            df_norm = df.divide(norm_flux)
                            Df = pd.concat([Df, df_norm], axis=0, join='outer')
                            norm_here = pd.DataFrame(data=[norm_flux],index=spec_id)
                            norm = pd.concat([norm, norm_here], axis=0, join='outer')
                            file.flush()
                            m += 1
                            if m % 1000 == 0:
                                plot_example_spectrum(path=str(output_directory_plots)+str(version)+'/',loglam=loglam,flux=flux,err=err,loglam_smooth=loglam_target,flux_smooth=int_flux,spec_id=spec_id[0],lam_in=lam_in,lam_out=lam_out,norm_flux=norm_flux,snr=snr)
                        df = []
                        df_norm = []
                        norm_here = []
                        file.write('spectrum processed\n')
                        file.flush()
                    except:
                        spec.close()
                        file.write('other issue, closing spectrum\n')
                        file.flush()
        except:
            file.write('spectrum not found!')

    file.write('shape after processing:')
    file.write(str(Df.shape)+'\n')
    file.write(str(norm.shape)+'\n')
    file.flush()
    if output_spectra_file != None:
        Df.to_csv(str(output_spectra_file))
    if output_norm_file != None:
        norm.to_csv(str(output_norm_file))
    file.close()

    return Df, norm

def perform_flux_cut(Df, output_file=None,thres_lam=1280):
    # Performs a set of cuts on the smoothed flux:
    # 1. Reject spectra whose fluxes fall below 0.5 below 128 nm (quasars with strongest associated absorption)
    # 2. Reject spectra whose fluxes fall below 0.1 above 128 nm (remove remaining quasars with poor SN ratio on the red side)
 
    thres_loglam = np.around(np.log10(thres_lam),decimals=4)
    thres_ind = Df.columns.get_loc(str(thres_loglam))
    Df_new = Df.drop(Df[Df.values[:,1:thres_ind]<0.5].index)
    Df_new = Df_new.drop(Df_new[Df_new.values[:,thres_ind:]<0.1].index)

    print('shape after flux cuts:')
    print(Df_new.shape)
    if output_file != None:
        Df_new.to_csv(str(output_file))

    return Df_new

def prepare_training_set(input_spectra_file,input_norm_file,output_spectra_file,output_norm_file,lam_in,lam_out):
    # Prepares the final training set, to be used for a Random Forest based cleanup.
    # The wavelength range is given by lam_start and lam_stop.

    Df_raw = input_spectra_file
    loglam_in_start = np.around(np.log10(lam_in[0]),decimals=4)
    loglam_in_stop = np.around(np.log10(lam_in[1]),decimals=4)
    loglam_out_start = np.around(np.log10(lam_out[0]),decimals=4)
    loglam_out_stop = np.around(np.log10(lam_out[1]),decimals=4)
    in_start = Df_raw.columns.get_loc(str(loglam_in_start))
    in_stop = Df_raw.columns.get_loc(str(loglam_in_stop))
    out_start = Df_raw.columns.get_loc(str(loglam_out_start))
    out_stop = Df_raw.columns.get_loc(str(loglam_out_stop))
    print('in start: ' + str(in_start))
    print('in stop: ' + str(in_stop))
    print('out start: ' + str(out_start))
    print('out stop: ' + str(out_stop))
    Df_clean = Df_raw.drop(Df_raw.columns[in_stop:],axis=1)
    Df_clean = Df_clean.drop(Df_clean.columns[1:out_start],axis=1)
    print(out_stop)
    print(in_start)
    if out_stop < in_start:
        Df_clean = Df_clean.drop(Df_clean.columns[out_stop:in_start],axis=1)
    # Drop quasars with missing data in this wavelength range.
    Df = Df_clean.dropna(axis=0)
    print(np.shape(Df))
    Df.to_csv(str(output_spectra_file))
    id_dat = input_norm_file
    print(id_dat.iloc[:,0])
    id_df = id_dat[id_dat.iloc[:,0].isin(Df.iloc[:,0])]
    id_df.to_csv(str(output_norm_file))

    return Df, id_df

def plot_example_spectrum(path,loglam,flux,err,loglam_smooth,flux_smooth,spec_id,lam_in,lam_out,norm_flux=None,snr=None):
    fig, axs = plt.subplots(2,1,sharey=True,figsize=(6.97,3.31))
    if norm_flux == None:
        axs[0].plot(10**loglam,flux,c=(0.25,0.25,0.25),label=r'${\rm ' +str(spec_id)+'\ raw\ data,\ SNR\ =\ ' + str(snr)+ '}$')
        axs[1].plot(10**loglam,flux,c=(0.25,0.25,0.25))
        axs[0].plot(10**loglam,1/np.sqrt(err),c='r',linewidth=1,label=r'${\rm \ flux\ errors}$')
        axs[1].plot(10**loglam,1/np.sqrt(err),c='r',linewidth=1)
        axs[0].plot(10**loglam_smooth,flux_smooth,color='c',label=r'${\rm QSmooth\ fit}$')
        axs[1].plot(10**loglam_smooth,flux_smooth,color='c')
        axs[0].set_ylabel(r'${\rm flux\ [erg\ s}$'+r'$^{-1} {\rm cm} ^{-2}\mathrm{\AA}$'+r'${\rm ]}$')
        axs[1].set_ylabel(r'${\rm flux\ [erg\ s}$'+r'$^{-1} {\rm cm} ^{-2}\mathrm{\AA}$'+r'${\rm ]}$')
        axs[0].set_ylim(bottom=-5)
        axs[1].set_ylim(bottom=-5)
    else:
        axs[0].plot(10**loglam,np.divide(flux,norm_flux),c=(0.25,0.25,0.25),label=r'${\rm ' +str(spec_id)+'\ raw\ data,\ SNR\ =\ ' + str(snr)+ '}$')
        axs[1].plot(10**loglam,np.divide(flux,norm_flux),c=(0.25,0.25,0.25))
        axs[0].plot(10**loglam,np.divide(1/np.sqrt(err),norm_flux),c='r',linewidth=1,label=r'${\rm \ flux\ errors}$')
        axs[1].plot(10**loglam,np.divide(1/np.sqrt(err),norm_flux),c='r',linewidth=1)
        axs[0].plot(10**loglam_smooth,np.divide(flux_smooth,norm_flux),color='c',label=r'${\rm QSmooth\ fit}$')
        axs[1].plot(10**loglam_smooth,np.divide(flux_smooth,norm_flux),color='c')
        axs[0].set_ylabel(r'${\rm normalized\ flux}$')
        axs[1].set_ylabel(r'${\rm normalized\ flux}$')
        if np.max(np.divide(flux,norm_flux)) > 15.0:
            axs[0].set_ylim([-0.02,15])
            axs[1].set_ylim([-0.02,15])
        elif np.max(np.divide(flux,norm_flux)) <= 15.0:
            axs[0].set_ylim([-0.02,1.3*np.max(np.divide(flux,norm_flux))])
            axs[1].set_ylim([-0.02,1.3*np.max(np.divide(flux,norm_flux))])
    axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
    axs[0].set_xlim([lam_out[0], lam_in[1]])
    axs[1].set_xlim([lam_out[0], lam_out[1]])
    axs[0].legend(frameon=False)
    plt.savefig(str(path)+str(spec_id)+'_example.png',bbox_inches='tight',dpi=300)
    plt.close()

def RF_cleanup(spectra_file,norms_file,path_to_clean_spectra,path_to_bad_spectra,path_to_clean_norms,path_to_bad_norms,RF_num=100,pca_variance=0.99,norm_lam=1290,n_folds=10):
    spectra = pd.read_csv(str(spectra_file))
    norms = pd.read_csv(str(norms_file))

    norm_loglam = np.around(np.log10(norm_lam),decimals=4)
    print(spectra.columns)

    spectra = spectra.drop(columns=str(norm_loglam)) # Drop 1290 column, as all spectra have flux normalized to 1 here (no variance)
    spectra = spectra.drop(spectra.columns[0:2],axis=1) # These contain row numbers and spectrum ids

    norms = norms.drop(norms.columns[0],axis=1) # This contains row numbers
    loglams = np.array(spectra.columns).astype(float) # Extract wavelengths

    # Divide spectra into input and output arrays
    blue = spectra.loc[:,:str(norm_loglam)].values
    red = spectra.loc[:,str(norm_loglam+0.0001):].values

    print(np.shape(blue))
    print(np.shape(red))
    print(np.shape(loglams))
    # K-fold split for cleanup
    skfolds = KFold(n_splits=n_folds, random_state=0, shuffle=True)
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
        test_blue_clean = np.where(((rf_blue_trans - test_blue_orig)/test_blue_orig) < err+5*std,test_blue_orig,np.nan)
        rf_blue_clean = np.where(((rf_blue_trans - test_blue_orig)/test_blue_orig) < err+5*std,rf_blue_trans,np.nan)

        # Calculate fraction of rejected data points
        rejected = sum(sum(np.isnan(rf_blue_clean)))
        total = int(np.size(rf_blue_clean))
        print('rejected '+str(float(rejected)/float(total)))
        print(np.shape(test_blue_clean))
        print(np.shape(test_red_orig))
        # Append the clean arrays to prepared dataframes
        test_new = np.concatenate((test_blue_clean,test_red_orig),axis=1)
        test_df = pd.DataFrame(data=test_new,columns=loglams,index=test_id)
        spectra_nan = pd.concat([spectra_nan, test_df], axis=0, join='outer')

        i += 1
    
    # Drop spectra with rejected data points
    spectra_clean = spectra_nan.dropna(axis=0,how='any')
    spectra_clean.to_csv(str(path_to_clean_spectra))
    norms_clean = norms.loc[np.array(spectra_clean.index)]
    norms_clean.to_csv(str(path_to_clean_norms))
	
    spectra_bad = spectra_nan[~spectra_nan.index.isin(spectra_clean.index)]
    spectra_bad.to_csv(str(path_to_bad_spectra))
    norms_bad = norms.loc[np.array(spectra_bad.index)]
    norms_bad.to_csv(str(path_to_bad_norms))

    return spectra_clean, norms_clean, spectra_bad, norms_bad

def stack_training_spectra(flag,version):
    if (flag == 'cleaned') or (flag == 'bad'):
        Df = pd.read_csv('data/dataframes/'+str(version)+'/training_spectra_'+str(flag)+'.csv')
        Df = Df.drop(Df.columns[0],axis=1)
        flux_ar = Df.values
        lam_ar = 10**Df.columns.values.astype(float)
        median_stack = np.nanmedian(flux_ar,axis=0)
        [p16, p84] = np.nanpercentile(flux_ar,[16,84],axis=0)

        plt.figure(figsize=(6.97,3.31))
        for j in range(0,100):
            plt.plot(lam_ar,flux_ar[j],color='gray',alpha=0.1)
        #plt.fill_between(lam_ar,p16,p84,color='coral',alpha=0.5,label=r'${\rm \ 16th-84th\ percentile}$')
        plt.plot(lam_ar,median_stack,color='brown',label=r'${\rm \ median\ stacked\ spectrum}$')
        plt.ylabel(r'${\rm normalized\ flux}$')
        plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
        plt.ylim([-0.1,1.2*np.max(np.nanpercentile(flux_ar,95,axis=0))])
        plt.xlim([1190, 1240])
        plt.legend(frameon=False)
        plt.savefig('plots/stack_'+str(flag)+'.png',bbox_inches='tight',dpi=300)
    else:
        print("invalid flag!")

def display_random_spectra(flag,lam_out,n=10,fits_dir='SDSS_quasars/',version='0'):
    if (flag == 'cleaned') or (flag == 'bad'):
        Df = pd.read_csv('data/dataframes/'+str(version)+'/training_spectra_'+str(flag)+'.csv')
        Df_norm = pd.read_csv('data/dataframes/'+str(version)+'/training_spectra_norms.csv')
        spec_ids = Df.iloc[:,0]
        Df = Df.drop(Df.columns[0],axis=1)
        flux_ar = Df.values
        lam_ar = 10**Df.columns.values.astype(float)
        rand_ids = randint(0,len(Df),n)

        fig, axs = plt.subplots(int(n/2),2,sharex=True,figsize=(6.97,n/2*1.00))
        plt.xlim([lam_out[0], lam_out[1]])
        for i in range(0,int(n/2)):
            filename = Df_norm.iloc[spec_ids[rand_ids[i]]][1]
            spec = fits.open(str(fits_dir)+str(filename))
            lam = 10**(spec[1].data['loglam'] - np.log10(1+spec[2].data['Z']))
            flux = spec[1].data['flux']/Df_norm.iloc[spec_ids[rand_ids[i]]][2]
            err = (1/np.sqrt(spec[1].data['ivar']))/Df_norm.iloc[spec_ids[rand_ids[i]]][2]
            spec.close()
            axs[i,0].plot(lam,flux,c='black',alpha=0.5)
            axs[i,0].plot(lam,err,c='gray',linewidth=0.5)
            axs[i,0].plot(lam_ar,flux_ar[rand_ids[i]],color='royalblue',label=str(filename))
            axs[i,0].set_ylim([-0.1,1.5*np.nanmax(flux_ar[rand_ids[i]])])
            axs[i,0].legend(frameon=False,loc='upper left')

            filename = Df_norm.iloc[spec_ids[rand_ids[int(n/2)+i]]][1]
            spec = fits.open(str(fits_dir)+str(filename))
            lam = 10**(spec[1].data['loglam'] - np.log10(1+spec[2].data['Z']))
            flux = spec[1].data['flux']/Df_norm.iloc[spec_ids[rand_ids[int(n/2)+i]]][2]
            err = (1/np.sqrt(spec[1].data['ivar']))/Df_norm.iloc[spec_ids[rand_ids[int(n/2)+i]]][2]
            spec.close()
            axs[i,1].plot(lam,flux,c='black',alpha=0.5)
            axs[i,1].plot(lam,err,c='gray',linewidth=0.5)
            axs[i,1].plot(lam_ar,flux_ar[rand_ids[int(n/2)+i]],color='royalblue',label=str(filename))
            axs[i,1].set_ylim([-0.1,1.5*np.nanmax(flux_ar[rand_ids[int(n/2)+i]])])
            axs[i,1].legend(frameon=False,loc='upper left')
        plt.subplots_adjust(wspace=0.1,hspace=0.0)
        plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
        plt.savefig('plots/random_'+str(flag)+'.png',bbox_inches='tight',dpi=300)
        plt.close()
    else:
        print("invalid flag!")
