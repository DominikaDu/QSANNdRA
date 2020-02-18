##########################################

# Author: Dominika Durovcikova (University of Oxford)
# Correspondence: dominika.durovcikova@gmail.com

# If used, please cite:

# Ďurovčíková, D., Katz, H., Bosman, S.E.I., Davies, F.B., Devriendt, J. and Slyz, A., 2019.
# Reionization history constraints from neural network based predictions of high-redshift quasar continua.
# arXiv preprint arXiv:1912.01050.

##########################################

# Structure

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
from QSmooth import open_fits, mask_SDSS, smooth

def perform_catalog_cut(catalog='DR14Q_v4_4.fits', z_start=2.09, z_end=2.51, out_path=None):
    # Performs a ZWARNING, BAL and redshift cut on the catalog given by catalog.

    hdu = pf.open(str(catalog_path))
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

    hdu_new = pf.BinTableHDU(data=hdudata_three)
    hdu_new.writeto(str(out_path)+'QSO_cut_1.fits')
    hdu.close()
    return hdu_new
    
def download_spec_files(catalog,target_directory):
    # Composes a download_SDSS.txt list of quasar spectra to be downloaded from the SAS via
    # wget -i download_SDSS.txt

    hdu = pf.open('QSO_cut.fits')
    hdudata = hdu[1].data
    # Create the txt file with writing permissions
    file = open('download_url.txt','w+')
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
    cmd = 'cd '+str(target_directory)
    os.system(cmd)
    cmd = 'wget -i download_SDSS.txt'
    os.system(cmd)

def preprocess_training_data(directory, SN_threshold):
    # Add description

    for filename in os.listdir(str(directory)):
        if filename.endswith('.fits'):
            m += 1 # counter
		    print(m)
		    print(str(filename))
		    spec = pf.open(str(directory)+str(filename))
            # Perform SN cut
            if spec[2].data['SN_MEDIAN_ALL'] < SN_threshold:
			    spec.close()
                print('closing spectrum')
		    else:
                mask = mask_SDSS(filename=filename,path=directory)
			    if len(data['loglam'][mask_ar])<1000:
				    spec.close()
				    print('closing spectrum')
                else:
                    try:
                        spec.close()
                        #need to calibrate the wavelengths acc to redshift here
                        [loglam_smooth, flux_smooth] = smooth(loglam,flux,err,mask=mask)
                        int_spec = interpolate.interp1d(loglam_smooth,flux_smooth,bounds_error=False,fill_value=np.nan) #interpolate the smooth spectrum
                        #set target wavelengths
                        start = loglam_smooth[0]
                        stop = loglam_smooth[-1]
                        if start >= norm_loglam: #if data points missing on the blue side
                            raise Exception('bad spectrum')
                        elif stop <= norm_loglam: #if data points missing on the red side
                            raise Exception('bad spectrum')
                        else:	
                            int_flux = int_spec(loglam_target) #flux values
                            #append these values to the overall dataframe
                            spec_id = [str(filename)]
                            df = pd.DataFrame(data=[int_flux],columns=loglam_target,index=spec_id)
                            norm_df = df.iloc[0].at[norm_loglam]
                            df_norm = df.divide(norm_df)
                            Df = Df.append(df_norm)
                            norm_here = pd.DataFrame(data=[norm_df],index=spec_id)
                            norm = norm.append(norm_here)
                        df = []
                        df_norm = []
                        norm_here = []
                    except:
                        spec.close()

##############################
			if len(data['loglam'][mask_ar])<1000:
				spec.close()
				print('closing spectrum')
			else:
				try:
					spec.close()
					[xx,yy] = smooth(filename,mask_ar)
					int_spec = interpolate.interp1d(xx,yy,bounds_error=False,fill_value=np.nan) #interpolate the smooth spectrum
					#set target wavelengths
					start = xx[0]
					stop = xx[-1]
					if start >= norm_loglam: #if data points missing on the blue side
						#shutil.move('../QSO_spectra/'+str(filename), '../QSO_spectra/QSO_bad/'+str(filename)) #move bad spectra to a subdirectory
						raise Exception('bad spectrum')
					elif stop <= norm_loglam: #if data points missing on the red side
						#shutil.move('../QSO_spectra/'+str(filename), '../QSO_spectra/QSO_bad/'+str(filename))
						raise Exception('bad spectrum')
					else:	
						int_flux = int_spec(loglam_target) #flux values
						#append these values to the overall dataframe
						spec_id = [str(filename)]
						df = pd.DataFrame(data=[int_flux],columns=loglam_target,index=spec_id)
						norm_df = df.iloc[0].at[norm_loglam]
						df_norm = df.divide(norm_df)
						Df = Df.append(df_norm)
						norm_here = pd.DataFrame(data=[norm_df],index=spec_id)
						norm = norm.append(norm_here)
					df = []
					df_norm = []
					norm_here = []
				except:
					spec.close()
	
#the full dataframe was created above
print('shape before 3rd set of cuts:')
print(Df.shape)
Df.to_csv('../dataframes/'+str(SN_treshold)+'SN_QSO_spectra_primary_dataframe_dense.csv')
norm.to_csv('../dataframes/'+str(SN_treshold)+'SN_QSO_spectra_norm_at_1290.csv')

#now the last cuts are the following:
#1. if the fluxes fall below 0.5 below 128 nm (remove quasars with strongest associated absorption)
#2. if the fluxes fall below 0.1 above 128 nm (remove remaining quasars with poor SN ratio on the red side)
 
norm_in = Df.columns.get_loc(norm_loglam)
Df_new = Df.drop(Df[Df.values[:,1:norm_in]<0.5].index)
Df_new = Df_new.drop(Df_new[Df_new.values[:,norm_in:]<0.1].index)

print('shape after 3rd set of cuts:')
print(Df_new.shape)
Df_new.to_csv('../dataframes/'+str(SN_treshold)+'SN_QSO_spectra_dataframe.csv')