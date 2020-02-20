# -*- coding: utf-8 -*-

import os
from scipy import interpolate
import pyfits as pf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn import linear_model
import preprocessing as pr

# Download and preprocess training data according to Section 2.1 in Ďurovčíková et al. 2019.

catalog_cut = pr.perform_catalog_cut(catalog='data/QSO_catalog/DR14Q_v4_4.fits', z_start=2.09, z_end=2.51)
pr.download_spec_files(hdu=catalog_cut,target_directory='data/training_fits/')
Df, norm_df = pr.preprocess_training_data(input_directory='data/training_fits/', SN_threshold=7.0, norm_lam = 1290, lam_start=1000, lam_stop=3900)
Df_new = pr.perform_flux_cut(Df=Df, thres_lam=1280)
df, ID_df = pr.prepare_training_set(input_spectra_file=Df_new,input_norm_file=norm_df,output_spectra_file='data/dataframes/training_spectra.csv',output_norm_file='data/dataframes/training_spectra_norms.csv',lam_start=1191.5,lam_stop=2900)
Df = []
Df_new = []
norm_df = []

# Refine the training set with Random Forest predictions as described in Section 2.2 in Ďurovčíková et al. 2019.

