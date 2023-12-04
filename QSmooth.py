# -*- coding: utf-8 -*-
##########################################

# Author: Dominika Ďurovčíková (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# D. Ďurovčíková, H. Katz, S. E. I. Bosman, F. B. Davies, J. Devriendt, and A. Slyz, Monthly Notices of the Royal Astronomical Society 493, 4256 (2020).

##########################################

import pyfits as pf
from scipy import interpolate
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn import linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def open_calibrate_fits(filename,path):
    hdu_raw = pf.open(str(path)+str(filename))
    # Either calibrate to rest wavelengths based on redshift from a particular line, e.g. Mg-II
    # https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/PLATE4/RUN1D/spZline.html
	# loglam = hdu['loglam'] - np.log10(1+hdu_raw[3].data['LINEZ'][5])
    # Or calibrate to rest wavelengths based on the redshift from the SDSS pipelines.
    loglam = hdu_raw[1].data['loglam'] - np.log10(1+hdu_raw[2].data['Z'])
    flux = hdu_raw[1].data['flux']
    err = hdu_raw[1].data['ivar']
    hdu_raw.close()
    return loglam, flux, err

def mask_SDSS(filename,path=None):
    # Opens an SDSS spec-PLATE-MJD-FIBER.fits file and creates a mask rejecting SDSS sky lines (Table 30 of Stoughton et al. 2002
    # https://ui.adsabs.harvard.edu/abs/2002AJ....123..485S/abstract) and data points flagged by SDSS pipelines.

    spec = pf.open(str(path)+str(filename))
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

def running_median(datx,daty,bin_size=30,shuffle=5,Lya=False):
    # Creates a running median of data given by datx and daty.
    # bin_size = number of data points to calculate one median value from
    # shuffle = step size as the number of data points to shuffle over
    # Lya = boolean parameter to reduce the bin size and shuffle in the vicinity of Lya peak

    j = 0
    xvals = []
    yvals = []
    if Lya:
        while True:
            if j+bin_size < len(datx):
                if (datx[j]>np.log10(1170)) & (datx[j]<np.log10(1270)): # if near Lya
                    j += bin_size/2 +shuffle/5
                    while datx[j+bin_size/5]<=np.log10(1270):
                        bin_x = np.mean(datx[j:j+bin_size/5])
                        bin_y = np.median(daty[j:j+bin_size/5])
                        j += shuffle/5
                        xvals.append(bin_x)
                        yvals.append(bin_y)
                else:
                    bin_x = np.mean(datx[j:j+bin_size])
                    bin_y = np.median(daty[j:j+bin_size])
                    j += shuffle
                    xvals.append(bin_x)
                    yvals.append(bin_y)
            else:
                shuffle = len(datx) - j
                bin_x = np.mean(datx[j:j+bin_size])
                bin_y = np.median(daty[j:j+bin_size])
                xvals.append(bin_x)
                yvals.append(bin_y)
                break
    else:
        while True:
            if j+bin_size < len(datx):
                bin_x = np.mean(datx[j:j+bin_size])
                bin_y = np.median(daty[j:j+bin_size])
                j += shuffle
                xvals.append(bin_x)
                yvals.append(bin_y)
            else:
                shuffle = len(datx) - j
                bin_x = np.mean(datx[j:j+bin_size])
                bin_y = np.median(daty[j:j+bin_size])
                xvals.append(bin_x)
                yvals.append(bin_y)
                break             
    return np.array(xvals),np.array(yvals)

def smooth(x,y,y_err,mask=[],bin_s=20,shuf=10,Lya=True):
    # Smooths raw input spectral data given by (x,y) and errors y_err according to procedure outlined in Appendix B
    # of Ďurovčíková et al. 2019 (https://arxiv.org/abs/1912.01050).
    # In this process, a mask can be used to reject some data points from the smoothing procedure.

    if len(mask)>0:
        x = x[mask]
        y = y[mask]
        y_err = y_err[mask]

    # 1. Compute upper envelope:
    [x_bor,y_bor] = running_median(x,y,bin_size=50,shuffle=10)
    border = interpolate.interp1d(x_bor,y_bor,bounds_error=False,fill_value='extrapolate')
    env_mask, _ = find_peaks(y,height=(border(x),None))
    x_env = x[env_mask]
    env = y[env_mask]
    f = interpolate.interp1d(x_env,env,bounds_error=False,fill_value='extrapolate')
    # 2. Subtract the envelope from raw data points to linearize the data:
    linear_y = y - f(x)
    linear_y = np.nan_to_num(linear_y)
    y_err[np.isnan(y_err)]=0.5 # Sufficiently small weight
	# 3. Apply RANSAC to detect outlying pixels (absorption features) and mask them out.
	# Note: we weigh the raw data points according to their errors.
    mad = np.average(np.abs(np.median(linear_y)-linear_y),weights=np.divide(y_err,np.sum(y_err)))
    ransac = linear_model.RANSACRegressor(random_state=0,loss='absolute_loss',residual_threshold=2.0*mad)
    ransac.fit(x.reshape(len(x),1), linear_y,sample_weight=np.abs(y_err))
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    #4. smooth the inlier data points
    [xx,yy] = running_median(x[inlier_mask],y[inlier_mask],bin_size=bin_s,shuffle=shuf,Lya=Lya)
    return np.array(xx), np.array(yy)
