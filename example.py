import os
from scipy import interpolate
import pyfits as pf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn import linear_model
import preprocessing as pr

_ = pr.perform_catalog_cut(catalog='data/QSO_catalog/DR14Q_v4_4.fits', z_start=2.09, z_end=2.51, out_file='data/dataframes/QSO_catalog_cut.fits')
pr.download_spec_files(filename='data/dataframes/QSO_catalog_cut.fits',target_directory='data/training_fits/')
Df, _ = pr.preprocess_training_data(input_directory='data/training_fits/', output_directory='data/dataframes/', SN_threshold=7.0, norm_lam = 1290, lam_start=1000, lam_stop=3900)
_ = pr.perform_flux_cut(Df=Df, output_directory='data/dataframes/', thres_lam=1280)