# QSANNdRA

This code was developed by Dominika Ďurovčíková and first implemented in [Ďurovčíková et al. 2020](https://academic.oup.com/mnras/article-abstract/493/3/4256/5741730?redirectedFrom=fulltext).

Please include the following citation if you use this code:

Ďurovčíková, D., Katz, H., Bosman, S.E.I., Davies, F.B., Devriendt, J. and Slyz, A., 2020.
Reionization history constraints from neural network based predictions of high-redshift quasar continua.
Monthly Notices of the Royal Astronomical Society, Volume 493, Issue 3, April 2020, Pages 4256–4275.

## Description

### Preprocessing

To preprocess low-redshift training data, use script "example_preprocessing.py".

![preprocessing example](plots/preprocessing/spec-4535-55860-0304.fits_example.png)

### Building QSANNdRA

To build and train QSANNdRA, use script "example_training.py".

### Application to high-redshift quasars

To preprocess high-redshift data, use script "example_high-z.py".

To apply QSANNdRA, use script "example_apply.py".

![example result](plots/high-z/example_result.png)

## Contact

Please contact Dominika Ďurovčíková at dominika.durovcikova@gmail.com in case of questions/issues.
