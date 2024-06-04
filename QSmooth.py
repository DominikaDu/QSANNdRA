# -*- coding: utf-8 -*-
##########################################

# Author: Dominika Ďurovčíková (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# Ďurovčíková, D., Katz, H., Bosman, S.E.I., Davies, F.B., Devriendt, J. and Slyz, A., 2019.
# Reionization history constraints from neural network based predictions of high-redshift quasar continua.
# arXiv preprint arXiv:1912.01050.

##########################################

import os
from astropy.io import fits
from scipy import interpolate
import pandas as pd
import numpy as np
from csaps import csaps
from scipy.signal import find_peaks
from sklearn import linear_model
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)

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
            if int(j+bin_size) < len(datx):
                if (datx[j]>np.log10(1170)) & (datx[j]<np.log10(1270)): # if near Lya
                    j += int(bin_size/2) +int(shuffle/5)
                    while datx[j+int(bin_size/5)]<=np.log10(1270):
                        bin_x = np.mean(datx[j:j+int(bin_size/5)])
                        bin_y = np.median(daty[j:j+int(bin_size/5)])
                        j += int(shuffle/5)
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

def ivarsmooth(flux, ivar, window):
    # Christina's code

    nflux = (flux.shape)[0]
    halfwindow = int(np.floor((np.round(window) - 1) / 2))
    shiftarr = np.zeros((nflux, 2 * halfwindow + 1))
    shiftivar = np.zeros((nflux, 2 * halfwindow + 1))
    shiftindex = np.zeros((nflux, 2 * halfwindow + 1))
    indexarr = np.arange(nflux)
    indnorm = np.outer(indexarr, (np.zeros(2 * halfwindow + 1) + 1))
    for i in np.arange(-halfwindow, halfwindow + 1, dtype=int):
        shiftarr[:, i + halfwindow] = np.roll(flux, i)
        shiftivar[:, i + halfwindow] = np.roll(ivar, i)
        shiftindex[:, i + halfwindow] = np.roll(indexarr, i)
    wh = (np.abs(shiftindex - indnorm) > (halfwindow + 1))
    shiftivar[wh] = 0.0
    outivar = np.sum(shiftivar, axis=1)
    nzero, = np.where(outivar > 0.0)
    zeroct = len(nzero)
    smoothflux = np.sum(shiftarr * shiftivar, axis=1)
    if (zeroct > 0):
        smoothflux[nzero] = smoothflux[nzero] / outivar[nzero]
    else:
        smoothflux = np.roll(flux, 2 * halfwindow + 1)  # kill off NAN's

    return smoothflux, outivar

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


def spline_smooth(x,y,y_err,smooth_b,smooth_r,thr=2.0,mask=None):
    # Smooths raw input spectral data given by (x,y) and errors y_err according to procedure outlined in Appendix B
    # of Ďurovčíková et al. 2019 (https://arxiv.org/abs/1912.01050).
    # In this process, a mask can be used to reject some data points from the smoothing procedure.

    # y_err is the std, not ivar!!

    if mask != None:
        x = x[mask]
        y = y[mask]
        y_err = y_err[mask]

    y_err[np.isinf(y_err)]=np.nan

    # bad pixels
    mask_err = np.logical_or(y_err < 2*np.nanstd(y_err),~np.isnan(y_err))
    x = x[mask_err]
    y = y[mask_err]
    y_err = y_err[mask_err]

    # 1. Compute upper envelope:
    [x_bor,y_bor] = running_median(x,y,bin_size=50,shuffle=10,Lya=True)
    border = interpolate.interp1d(x_bor,y_bor,bounds_error=False,fill_value='extrapolate')
    env_mask, _ = find_peaks(y,height=(border(x),None))
    x_env = x[env_mask]
    env = y[env_mask]
    f = interpolate.interp1d(x_env,env,bounds_error=False,fill_value='extrapolate')
    # 2. Subtract the envelope from raw data points to linearize the data:
    linear_y = y - f(x)
    linear_y = np.nan_to_num(linear_y)
	# 3. Apply RANSAC to detect outlying pixels (absorption features) and mask them out.
	# Note: we weigh the raw data points according to their errors.
    w = np.abs(1/y_err)
    w[np.isinf(w)]=1e6
    mad = np.average(np.abs(np.median(linear_y)-linear_y),weights=w)
    ransac = linear_model.RANSACRegressor(random_state=0,loss='absolute_error',residual_threshold=thr*mad)
    ransac.fit(x.reshape(len(x),1), linear_y,sample_weight=w)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    normid = np.abs(x[inlier_mask] - 3.1106).argmin()+1
    normid_all = np.abs(x - 3.1106).argmin()+1
    smooth_blue = csaps(x[inlier_mask][:normid], y[inlier_mask][:normid], x[:normid_all], weights=w[inlier_mask][:normid], smooth=smooth_b, normalizedsmooth=False)
    smooth_red = csaps(x[inlier_mask][normid:], y[inlier_mask][normid:], x[normid_all:], weights=w[inlier_mask][normid:], smooth=smooth_r, normalizedsmooth=False)
    spl = interpolate.interp1d(x,np.concatenate((smooth_blue,smooth_red)),bounds_error=False,fill_value='extrapolate')

    return spl, outlier_mask

def view_smooth_spectrum(path,lam,flux,err,specname,mask,mask_smooth,outlier_mask,lam_smooth,flux_smooth,norm_flux=None,xlim=[1150,3000],ylimtop=10.0,zoom=False,legend=True,lines=False,z=None,tells=[13500.0,14200.0,18000.0,19400.0,24260.0]):
    mlam = np.ma.masked_array(data=lam,mask=~mask).filled(np.nan)
    mflux = np.ma.masked_array(data=flux,mask=~mask).filled(np.nan)
    merr = np.ma.masked_array(data=err,mask=~mask).filled(np.nan)
    xcoords = np.concatenate((lam_smooth[~mask_smooth],lam_smooth[mask_smooth][outlier_mask]))

    fluxint = interpolate.interp1d(lam_smooth,flux_smooth,bounds_error=False,fill_value='extrapolate')

    if lines == True:
        fig, axs = plt.subplot_mosaic([['full spectrum','full spectrum','full spectrum','full spectrum','full spectrum'],\
            ['full spectrum','full spectrum','full spectrum','full spectrum','full spectrum'],\
                [r'${\rm Lya+NV}$',r'${\rm CII+SiIV}$',r'${\rm CIV}$',r'${\rm CIII}$',r'${\rm MgII}$']],constrained_layout=True,figsize=(6.97,3.31))
        if norm_flux == None:
            for label, ax in axs.items():
                if label == 'full spectrum':
                    ax.plot(lam,flux,c=(0.25,0.25,0.25),label=r'${\rm ' +str(specname)+'\ raw\ data}$')
                    ax.plot(lam,err,c='r',linewidth=0.3,label=r'${\rm \ flux\ errors}$')
                    for xc in xcoords:
                        ax.axvline(x=xc, c='lightgray',alpha=0.05)
                    ax.axvspan(tells[0]/(1+z),tells[1]/(1+z),color='lightgray',zorder=2)
                    ax.axvspan(tells[2]/(1+z),tells[3]/(1+z),color='lightgray',zorder=2)
                    if tells[4]/(1+z) < 2900:
                        ax.axvspan(tells[4]/(1+z),2900,color='lightgray',zorder=2)
                    ax.plot(lam_smooth,flux_smooth,color='c',label=r'${\rm QSmooth\ fit}$')
                    ax.set_ylabel(r'${{\rm flux\ [ } 10^{-17} {\rm erg\ s}}$'+r'$^{-1} {\rm cm} ^{-2}\mathrm{\AA}^{-1}$'+r'${\rm ]}$')
                    ax.set_ylim([-0.02,ylimtop])
                    ax.set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
                    ax.set_xlim([1150, 2900])
                    ax.legend(frameon=False,loc='upper right')
                    if z != None:
                        secax = ax.secondary_xaxis('top', functions=(lambda x: x*(1+z), lambda x: x/(1+z)))
                        secax.set_xlabel(r'${\rm observed\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
                else:
                    ax.plot(lam,flux,c=(0.25,0.25,0.25))
                    ax.plot(lam,err,c='r',linewidth=0.3)
                    for xc in xcoords:
                        ax.axvline(x=xc, c='lightgray',alpha=0.05)
                    ax.axvspan(tells[0]/(1+z),tells[1]/(1+z),color='lightgray',zorder=2)
                    ax.axvspan(tells[2]/(1+z),tells[3]/(1+z),color='lightgray',zorder=2)
                    if tells[4]/(1+z) < 2900:
                        ax.axvspan(tells[4]/(1+z),2900,color='lightgray',zorder=2)
                    ax.plot(lam_smooth,flux_smooth,color='c')
                    ax.text(0.02,0.85,label,transform=ax.transAxes)
        else:
            for label, ax in axs.items():
                if label == 'full spectrum':
                    ax.plot(lam,flux,c=(0.25,0.25,0.25,0.3))
                    ax.plot(lam,err,c='r',alpha=0.3,linewidth=1)
                    ax.plot(mlam,np.divide(mflux,norm_flux),c=(0.25,0.25,0.25),label=r'${\rm ' +str(specname)+'\ raw\ data}$')
                    ax.plot(lam_smooth,flux_smooth,color='c',label=r'${\rm QSmooth\ fit}$')
                    ax.plot(mlam,np.divide(merr,norm_flux),c='r',linewidth=1,label=r'${\rm \ flux\ errors}$')
                    ax.ylabel(r'${\rm normalized\ flux}$')
                    ax.ylim([-0.02,ylimtop])
                    ax.set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
                    ax.set_xlim([1150, 2900])
                    ax.legend(frameon=False,loc='upper right')
                else:
                    ax.plot(lam,flux,c=(0.25,0.25,0.25,0.3))
                    ax.plot(lam,err,c='r',alpha=0.3,linewidth=1)
                    ax.plot(mlam,np.divide(mflux,norm_flux),c=(0.25,0.25,0.25))
                    ax.plot(lam_smooth,flux_smooth,color='c')
                    ax.plot(mlam,np.divide(merr,norm_flux),c='r',linewidth=1)
                    ax.ylim([-0.02,ylimtop])
        axs[r'${\rm Lya+NV}$'].set_ylim([-0.02,3.0*fluxint(1250)])
        #axs[r'${\rm OI+CII+SiIV+OIV}$'].set_ylim([-0.02,2.0*fluxint(1400)])
        axs[r'${\rm CII+SiIV}$'].set_ylim([-0.02,2.0*fluxint(1400)])
        axs[r'${\rm CIV}$'].set_ylim([-0.02,2.0*fluxint(1550)])
        axs[r'${\rm CIII}$'].set_ylim([-0.02,2.0*fluxint(1900)])
        axs[r'${\rm MgII}$'].set_ylim([-0.02,2.0*fluxint(2800)])
        axs[r'${\rm Lya+NV}$'].set_xlim([1180,1280])
        axs[r'${\rm CII+SiIV}$'].set_xlim([1280,1450])
        axs[r'${\rm CIV}$'].set_xlim([1450,1600])
        axs[r'${\rm CIII}$'].set_xlim([1860,1960])
        axs[r'${\rm MgII}$'].set_xlim([2750,2850])
        plt.savefig(str(path)+str(specname)+'_smooth+lines.png',bbox_inches='tight',dpi=300)
        plt.close()

    if lines == False:
        if zoom == False:
            plt.figure(figsize=(6.97,2.00))
            if norm_flux == None:
                plt.plot(lam,flux,c=(0.25,0.25,0.25),label=r'${\rm ' +str(specname)+'\ raw\ data}$')
                plt.plot(lam,err,c='r',linewidth=0.3,label=r'${\rm \ flux\ errors}$')
                for xc in xcoords:
                    plt.axvline(x=xc, c='lightgray',alpha=0.1)
                plt.plot(lam_smooth,flux_smooth,color='c',label=r'${\rm QSmooth\ fit}$')
                plt.ylabel(r'${{\rm flux\ [ } 10^{-17} {\rm erg\ s}}$'+r'$^{-1} {\rm cm} ^{-2}\mathrm{\AA}^{-1}$'+r'${\rm ]}$')
                plt.ylim([-0.02,ylimtop])
            else:
                plt.plot(lam,flux,c=(0.25,0.25,0.25,0.3))
                plt.plot(lam,err,c='r',alpha=0.3,linewidth=1)
                plt.plot(mlam,np.divide(mflux,norm_flux),c=(0.25,0.25,0.25),label=r'${\rm ' +str(specname)+'\ raw\ data}$')
                plt.plot(lam_smooth,flux_smooth,color='c',label=r'${\rm QSmooth\ fit}$')
                plt.plot(mlam,np.divide(merr,norm_flux),c='r',linewidth=1,label=r'${\rm \ flux\ errors}$')
                plt.ylabel(r'${\rm normalized\ flux}$')
                plt.ylim([-0.02,ylimtop])
            plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
            plt.xlim([1150, 2900])
            if legend == True:
                plt.legend(frameon=False,loc='upper right')
            plt.savefig(str(path)+str(specname)+'_smooth.png',bbox_inches='tight',dpi=300)
            plt.close()
            
        elif zoom == True:
            fig, axs = plt.subplots(2,1,sharey=True,figsize=(6.97,3.31))
            if norm_flux == None:
                axs[0].plot(lam,flux,c=(0.25,0.25,0.25),label=r'${\rm ' +str(specname)+'\ raw\ data}$')
                axs[1].plot(lam,flux,c=(0.25,0.25,0.25))
                axs[0].plot(lam,err,c='r',linewidth=0.3,label=r'${\rm \ flux\ errors}$')
                axs[1].plot(lam,err,c='r',linewidth=0.3)
                for xc in xcoords:
                    axs[0].axvline(x=xc, c='lightgray',alpha=0.1)
                    axs[1].axvline(x=xc, c='lightgray',alpha=0.1)
                #axs[0].plot(mlam,mflux,c=(0.25,0.25,0.25),label=r'${\rm ' +str(specname)+'\ raw\ data}$')
                #axs[1].plot(mlam,mflux,c=(0.25,0.25,0.25))
                axs[0].plot(lam_smooth,flux_smooth,color='c',label=r'${\rm QSmooth\ fit}$')
                axs[1].plot(lam_smooth,flux_smooth,color='c')
                #axs[0].plot(mlam,merr,c='r',linewidth=1,label=r'${\rm \ flux\ errors}$')
                #axs[1].plot(mlam,merr,c='r',linewidth=1)
                axs[0].set_ylabel(r'${{\rm flux\ [ } 10^{-17} {\rm erg\ s}}$'+r'$^{-1} {\rm cm} ^{-2}\mathrm{\AA}^{-1}$'+r'${\rm ]}$')
                axs[1].set_ylabel(r'${{\rm flux\ [ } 10^{-17} {\rm erg\ s}}$'+r'$^{-1} {\rm cm} ^{-2}\mathrm{\AA}^{-1}$'+r'${\rm ]}$')
                axs[0].set_ylim([-0.02,ylimtop])
                axs[1].set_ylim([-0.02,ylimtop])
            else:
                axs[0].plot(lam,flux,c=(0.25,0.25,0.25,0.3))
                axs[1].plot(lam,flux,c=(0.25,0.25,0.25,0.3))
                axs[0].plot(lam,err,c='r',alpha=0.3,linewidth=1)
                axs[1].plot(lam,err,c='r',alpha=0.3,linewidth=1)
                axs[0].plot(mlam,np.divide(mflux,norm_flux),c=(0.25,0.25,0.25),label=r'${\rm ' +str(specname)+'\ raw\ data}$')
                axs[1].plot(mlam,np.divide(mflux,norm_flux),c=(0.25,0.25,0.25))
                axs[0].plot(lam_smooth,flux_smooth,color='c',label=r'${\rm QSmooth\ fit}$')
                axs[1].plot(lam_smooth,flux_smooth,color='c')
                axs[0].plot(mlam,np.divide(merr,norm_flux),c='r',linewidth=1,label=r'${\rm \ flux\ errors}$')
                axs[1].plot(mlam,np.divide(merr,norm_flux),c='r',linewidth=1)
                axs[0].set_ylabel(r'${\rm normalized\ flux}$')
                axs[1].set_ylabel(r'${\rm normalized\ flux}$')
                axs[0].set_ylim([-0.02,ylimtop])
                axs[1].set_ylim([-0.02,ylimtop])
            axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
            axs[0].set_xlim([1150, 2900])
            axs[1].set_xlim([1150, 1500])
            axs[0].legend(frameon=False,loc='upper right')
            plt.savefig(str(path)+str(specname)+'_smooth.png',bbox_inches='tight',dpi=300)
            plt.close()

def prepare_spectrum_forQ(specname,sf,fint,norm_lam,lam_start,lam_stop):
        norm_loglam = np.around(np.log10(norm_lam),decimals=4)
        loglam_start = np.around(np.log10(lam_start),decimals=4)
        loglam_stop = np.around(np.log10(lam_stop),decimals=4)    
        loglam_target = np.around(np.arange(loglam_start,loglam_stop,0.0001),decimals=4)

        norm = np.float(fint(norm_loglam))
        flux_int = np.divide(fint(loglam_target),norm)

        spec_norm = pd.DataFrame(data=[flux_int],columns=np.asarray(loglam_target,dtype='str'))
        
        np.savetxt(str(specname)+'/smooth_'+str(sf)+'.txt',spec_norm,delimiter=' ',fmt='%1.10f',\
            header='Reconstructed spectrum for '+str(specname)+' normalized at 1290 A. Normalization constant: '+str(norm)) # This spectrum is normalized wrt 1290 A!
