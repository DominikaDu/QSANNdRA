# QSANNdRA

This code was developed by Dominika Ďurovčíková and first implemented in [Ďurovčíková et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.4256D/abstract), and it was most recently updated as a result of [Ďurovčíková et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240110328D/abstract).

Please include the following citation if you use this code:

D. Ďurovčíková, H. Katz, S. E. I. Bosman, F. B. Davies, J. Devriendt, and A. Slyz, Monthly Notices of the Royal Astronomical Society 493, 4256 (2020).


## Update

As of June 2024, a Jupyter notebook tutorial has been added, see QSANNdRA-tutorial.ipynb. This notebook allows the user to specify the input and output wavelength ranges without requiring that Lyman alpha is covered by the output (this is for example relevant for Lyman alpha forest analyses). All other pre-existing files have been moved to archive/ and are not needed to run the tutorial.

## Contact

Please contact Dominika Ďurovčíková at dominika.durovcikova@gmail.com in case of questions/issues.

## Description of archival scripts

### Preprocessing

To preprocess low-redshift training data, use script "example_preprocessing.py".

![preprocessing example](plots/preprocessing/spec-4535-55860-0304.fits_example.png)

### Building QSANNdRA

To build and train QSANNdRA, use script "example_training.py".

### Application to high-redshift quasars

To preprocess high-redshift data, use script "example_high-z.py".

To apply QSANNdRA, use script "example_apply.py".

![example result](plots/high-z/example_result.png)