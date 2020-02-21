import numpy as np
import QSANNdRA as Q

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

training_spectra = 'data/dataframes/good/training_spectra_cleaned.csv'
training_norms = 'data/dataframes/good/training_spectra_norms_cleaned.csv'
path_to_models = 'models/'

_, _ = Q.get_QSANNdRA(training_spectra=training_spectra,training_norms=training_norms,models_save=True,path_to_models=path_to_models,C=100,load=False,save_errs=True)
