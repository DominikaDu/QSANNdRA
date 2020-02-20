import numpy as np
import QSANNdRA as Q

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

training_spectra = 'data/dataframes/good/training_spectra_cleaned.csv' # This file seems to have too few spectra! maybe SN 9?
training_norms = 'data/dataframes/good/training_spectra_norms_cleaned.csv'
path_to_models = 'models/preprocessing_models/'

scaler_red_one, pca_red, scaler_red_two, train_red, test_red, \
    scaler_blue_one, pca_blue, scaler_blue_two, train_blue, test_blue = \
        Q.prepare_training_data(training_spectra=training_spectra,training_norms=training_norms,models_save=True,path_to_models=path_to_models)

# train full committee

# inverse transform training sets and predictions

# evaluate performance
