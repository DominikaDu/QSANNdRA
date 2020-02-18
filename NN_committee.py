#creates a committee of C networks with different initialisation seed
C = 100
s = range(1,C+1)
red_comp = 63 #also number of neurons in input layer
blue_comp = 36 #also number of neurons in output layer
n_two = 40
n_three = 40
n_ep = 80 #number of epochs
batch_s = 500 #batch size
act = 'elu'

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import pandas as pd

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


import joblib

print('reading data...')
Df = pd.read_csv('../dataframes/QSO_spectra_cleaned_ids.csv')
IDs = pd.read_csv('../dataframes/QSO_spectra_norm_cleaned.csv')
Df = Df.drop(Df.columns[0],axis=1)
IDs = IDs.drop(IDs.columns[0],axis=1)

print('splitting data...')
#split data into train and test acc to ratio 80:20
train = Df.iloc[:(np.int(0.8*len(Df)))]
test = Df.iloc[(np.int(0.8*len(Df))):]

print('creating train and test sets...')
train_blue_orig = train.loc[:,:'3.1107'].values
train_red_orig = train.loc[:,'3.1107':].values
test_blue_orig = test.loc[:,:'3.1107'].values
test_red_orig = test.loc[:,'3.1107':].values

from sklearn.preprocessing import StandardScaler
scaler_red = StandardScaler()
scaler_red.fit(train_red_orig)
train_red = scaler_red.transform(train_red_orig)
test_red = scaler_red.transform(test_red_orig)
scaler_blue = StandardScaler()
scaler_blue.fit(train_blue_orig)
train_blue = scaler_blue.transform(train_blue_orig)
test_blue = scaler_blue.transform(test_blue_orig)
joblib.dump(scaler_red,'../trained_models/scaler_red.pkl')
joblib.dump(scaler_blue,'../trained_models/scaler_blue.pkl')


print('PCA')
from sklearn.decomposition import PCA
pca_red = PCA(n_components=red_comp)
pca_red.fit(train_red)
train_red = pca_red.transform(train_red)
test_red = pca_red.transform(test_red)
print('percentage of information preserved under PCA_red')
print(sum(pca_red.explained_variance_ratio_)) #how much information is contained within the components
pca_blue = PCA(n_components=blue_comp)
pca_blue.fit(train_blue)
train_blue = pca_blue.transform(train_blue)
test_blue = pca_blue.transform(test_blue)
print('percentage of information preserved under PCA_blue')
print(sum(pca_blue.explained_variance_ratio_))
joblib.dump(pca_red,'../trained_models/pca_red.pkl')
joblib.dump(pca_blue,'../trained_models/pca_blue.pkl')

scaler_red_two = StandardScaler()
scaler_red_two.fit(train_red)
train_red = scaler_red_two.transform(train_red)
test_red = scaler_red_two.transform(test_red)
scaler_blue_two = StandardScaler()
scaler_blue_two.fit(train_blue)
train_blue = scaler_blue_two.transform(train_blue)
test_blue = scaler_blue_two.transform(test_blue)
joblib.dump(scaler_red_two,'../trained_models/scaler_red_two.pkl')
joblib.dump(scaler_blue_two,'../trained_models/scaler_blue_two.pkl')

loglams_test = np.array(test.columns).astype(np.float) #extract columns headers into a np array

print('data is prepared')

#create the committee of NNs
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import initializers

predictions_train = np.empty((np.shape(train_blue_orig)[0],np.shape(train_blue_orig)[1],C))
predictions_test = np.empty((np.shape(test_blue_orig)[0],np.shape(test_blue_orig)[1],C))
errs = np.empty((np.shape(test_blue_orig)[1],C))
errs_tr = np.empty((np.shape(train_blue_orig)[1],C))

for i in range (0,C):
	print(i)
	#define architecture
	model = Sequential()
	model.add(Dense(n_two, input_dim=red_comp, kernel_initializer=initializers.RandomNormal(seed=s[i]),bias_initializer=initializers.RandomNormal(seed=s[i]),activation=act))
	model.add(Dense(n_three, kernel_initializer=initializers.RandomNormal(seed=s[i]),bias_initializer=initializers.RandomNormal(seed=s[i]),activation=act))
	model.add(Dense(blue_comp, kernel_initializer=initializers.RandomNormal(seed=s[i]),bias_initializer=initializers.RandomNormal(seed=s[i])))
	model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mae'])
	model.load_weights('../trained_models/NN_committee_'+str(i)+'.h5')

	#checkpointer = ModelCheckpoint(filepath='../trained_models/NN_committee_'+str(i)+'.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',period=5) #tells the NN when to stop training

	#history = model.fit(train_red, train_blue, epochs=n_ep, batch_size=batch_s,verbose=1, shuffle=True,validation_split=0.2,callbacks=[checkpointer])
	pred_blue = scaler_blue.inverse_transform(pca_blue.inverse_transform(scaler_blue_two.inverse_transform(model.predict(test_red))))
	pred_train_blue = scaler_blue.inverse_transform(pca_blue.inverse_transform(scaler_blue_two.inverse_transform(model.predict(train_red))))
	predictions_train[:,:,i] = pred_train_blue
	predictions_test[:,:,i] = pred_blue
	error = ((np.abs(pred_blue - test_blue_orig)/test_blue_orig)).mean(axis=0) #calculate mean absolute relative error for each wavelength
	np.savetxt('../trained_models/NN'+str(i)+'_errors.csv',np.transpose(error),delimiter=',')
	std = ((np.abs(pred_blue - test_blue_orig)/test_blue_orig)).std(axis=0)
	er_train = ((np.abs(pred_train_blue - train_blue_orig)/train_blue_orig)).mean(axis=0)
	std_train = ((np.abs(pred_train_blue - train_blue_orig)/train_blue_orig)).std(axis=0)
	bias_train = (((pred_train_blue - train_blue_orig)/train_blue_orig)).mean(axis=0)
	bias_test = (((pred_blue - test_blue_orig)/test_blue_orig)).mean(axis=0) #calculate mean absolute relative error for each wavelength
	np.savetxt('../trained_models/NN'+str(i)+'_stds.csv',np.transpose(std),delimiter=',')
	np.savetxt('../trained_models/NN'+str(i)+'_stds_train.csv',np.transpose(std_train),delimiter=',')
	np.savetxt('../trained_models/NN'+str(i)+'_errors_train.csv',np.transpose(er_train),delimiter=',')
	np.savetxt('../trained_models/NN'+str(i)+'_bias_train.csv',np.transpose(bias_train),delimiter=',')
	np.savetxt('../trained_models/NN'+str(i)+'_bias_test.csv',np.transpose(bias_test),delimiter=',')
	errs[:,i] = 1-error
	errs_tr[:,i] = 1-((np.abs(pred_train_blue - train_blue_orig)/train_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength


#take the weighted mean of all predictions for each wavelength for each spectrum
err_norm = errs.sum(axis=1)
w = np.divide(errs,err_norm[:,None])
err_norm_tr = errs_tr.sum(axis=1)
w_tr = np.divide(errs_tr,err_norm_tr[:,None])
w_test = np.repeat(w[np.newaxis,:, :], len(predictions_test), axis=0)
w_train = np.repeat(w_tr[np.newaxis,:,:],len(predictions_train),axis=0)

mean_preds_train = np.average(predictions_train,weights=w_train,axis=2)
mean_preds_test = np.average(predictions_test,weights=w_test,axis=2)

np.shape(mean_preds_train)
np.shape(mean_preds_test)

################################################
# Fred's test

# predictions_test = contains individual predictions (2741, 346, 100)
# mean_preds_test = contains mean predictions (2741, 346)
# want to measure scatter between individual predictions and the mean prediction

mae = np.zeros((346,100))
st = np.zeros((346,100))
for ii in range(0,100):
	mae[:,ii] = (np.abs(predictions_test[:,:,ii]-mean_preds_test)/mean_preds_test).mean(axis=0)
	st[:,ii] = (np.abs(predictions_test[:,:,ii]-mean_preds_test)/mean_preds_test).std(axis=0)

scatter = mae.mean(axis=1)
scatter_st = st.mean(axis=1)

plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
plt.figure(figsize=(3.31,3.31))
plt.plot(10**(loglams_test[:346]), scatter, c='mediumvioletred',label=r'${\rm mean\ scatter}$')
plt.plot(10**(loglams_test[:346]), scatter_st, c='mediumvioletred', linestyle='dashed',label=r'${\rm std\ of\ scatter}$')
plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
plt.ylabel(r'${\rm mean\ absolute\ error\ (scatter\ around\ mean) }$')
plt.savefig('../paper/scatter_NNs.png',bbox_inches='tight',dpi=400)


##############################

mae = np.zeros((2741,346,100))
for ii in range(0,100):
	mae[:,:,ii] = (np.abs(predictions_test[:,:,ii]-mean_preds_test)/mean_preds_test)

scatter = mae.mean(axis=2)
scatter_final = scatter.mean(axis=0)

plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
plt.figure(figsize=(3.31,3.31))
plt.plot(10**(loglams_test[:346]), scatter_final, c='mediumvioletred')
plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
plt.ylabel(r'${\rm mean\ absolute\ error\ (scatter\ around\ mean) }$')
plt.savefig('../paper/scatter_NNs_2.png',bbox_inches='tight',dpi=400)

################################################

proj_pred_tr = mean_preds_train
err_train = ((np.abs(proj_pred_tr - train_blue_orig)/train_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength
std_train = ((np.abs(proj_pred_tr - train_blue_orig)/train_blue_orig)).std(axis=0) #calculate std squared relative error for each wavelength

proj_pred_b = mean_preds_test
err_test = ((np.abs(proj_pred_b - test_blue_orig)/test_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength
std_test = ((np.abs(proj_pred_b - test_blue_orig)/test_blue_orig)).std(axis=0) #calculate std squared relative error for each wavelength
np.savetxt('error_test_committee.csv',np.transpose([err_test,std_test]),delimiter=',')
np.savetxt('error_train_committee.csv',np.transpose([err_train,std_train]),delimiter=',')

bias_train = (((proj_pred_tr - train_blue_orig)/train_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength
bias_test = (((proj_pred_b - test_blue_orig)/test_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength

err_per_datapoint = ((np.abs(proj_pred_b - test_blue_orig)/test_blue_orig)).mean()
print('mean absolute error per datapoint in the test set = '+str(err_per_datapoint))

#
#mean absolute error per datapoint in the test set = 0.05491413577884477
# 0.05440227946430592 with initialized biases
# 0.054892651971112455 (new)

b_err_train = (((proj_pred_tr - train_blue_orig)/train_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength
b_std_train = (((proj_pred_tr - train_blue_orig)/train_blue_orig)).std(axis=0) #calculate std squared relative error for each wavelength

b_err_test = (((proj_pred_b - test_blue_orig)/test_blue_orig)).mean(axis=0) #calculate mean squared relative error for each wavelength
b_std_test = (((proj_pred_b - test_blue_orig)/test_blue_orig)).std(axis=0) #calculate std squared relative error for each wavelength

###################################
one_train = np.loadtxt('error_train.csv',delimiter=',')
one_test = np.loadtxt('error_test.csv',delimiter=',')
one_b_train = np.loadtxt('bias_train.csv',delimiter=',')
one_b_test = np.loadtxt('bias_test.csv',delimiter=',')

from matplotlib.legend_handler import HandlerBase

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
	print(orig_handle)
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], color=(0.25,0.25,0.25),linestyle=orig_handle[0])
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color='mediumvioletred',linestyle=orig_handle[1])
        return [l1, l2]



plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
f,axs = plt.subplots(1,2,sharey=True,sharex=True,figsize=(6.97,3.31))
one = axs[0].plot(10**(loglams_test[:len(err_train)]),one_train[:,0],c=(0.25,0.25,0.25),linestyle='solid')
hundred = axs[0].plot(10**(loglams_test[:len(err_train)]),err_train,c='mediumvioletred',linestyle='solid')
one_std = axs[0].plot(10**(loglams_test[:len(err_train)]),one_train[:,1],c=(0.25,0.25,0.25),linestyle='--')
hundred_std = axs[0].plot(10**(loglams_test[:len(err_train)]),std_train,c='mediumvioletred',linestyle='--')
axs[0].plot(10**(loglams_test[:len(one_b_train)]),one_b_train[:,0],c=(0.25,0.25,0.25),linestyle='dotted')
axs[0].plot(10**(loglams_test[:len(b_err_train)]),b_err_train,c='mediumvioletred',linestyle='dotted')
axs[1].plot(10**(loglams_test[:len(err_test)]),one_test[:,0],c=(0.25,0.25,0.25),label='test set $\overline{\epsilon}$')
axs[1].plot(10**(loglams_test[:len(err_test)]),one_test[:,1],c=(0.25,0.25,0.25),linestyle='--',label='test set 1$\sigma$')
axs[1].plot(10**(loglams_test[:len(err_test)]),err_test,c='mediumvioletred')
axs[1].plot(10**(loglams_test[:len(err_test)]),std_test,c='mediumvioletred',linestyle='--')
axs[1].plot(10**(loglams_test[:len(one_b_test)]),one_b_test[:,0],c=(0.25,0.25,0.25),linestyle='dotted')
axs[1].plot(10**(loglams_test[:len(b_err_test)]),b_err_test,c='mediumvioletred',linestyle='dotted')
axs[0].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
axs[0].set_ylabel(r'${\rm relative\ prediction\ error}$')
plt.subplots_adjust(wspace=0, hspace=0.05)
axs[0].legend([('solid','solid'),('--','--'),('dotted','dotted')],[r'${\rm train\ set\ }$' r'$\overline{\epsilon}$',r'${\rm train\ set\ }$' r'$\sigma_{\epsilon}$',r'${\rm train\ set\ bias}$'],handler_map={tuple:AnyObjectHandler()},frameon=False)
axs[1].legend([('solid','solid'),('--','--'),('dotted','dotted')],[r'${\rm test\ set\ }$' r'$\overline{\epsilon}$',r'${\rm test\ set\ }$' r'$\sigma_{\epsilon}$',r'${\rm test\ set\ bias}$'],handler_map={tuple:AnyObjectHandler()},frameon=False)
#axs[0].legend([('solid','solid'),('--','--')],['train set $\overline{\epsilon}$','train set $\sigma_{\epsilon}$'],handler_map={tuple:AnyObjectHandler()})
#axs[1].legend([('solid','solid'),('--','--')],['test set $\overline{\epsilon}$','test set $\sigma_{\epsilon}$'],handler_map={tuple:AnyObjectHandler()})
f.legend([one,hundred],labels=[r'${\rm one\ neural\ network}$',r'${\rm committee\ of\ 100\ networks}$'],loc='upper center',frameon=False,borderpad=0.0,bbox_to_anchor=(0.15,0.73,0.25,0.25),borderaxespad=0,mode='expand')
plt.show()
plt.savefig('../paper/one_to_hundred_comp.png',bbox_inches='tight',dpi=400)
plt.savefig('../paper/one_to_hundred_comp.pdf',bbox_inches='tight',dpi=400)


best_er_test = np.loadtxt('../trained_models/NN95_errors.csv',delimiter=',')
best_std_test = np.loadtxt('../trained_models/NN95_stds.csv',delimiter=',')
best_er_train = np.loadtxt('../trained_models/NN95_errors_train.csv',delimiter=',')
best_std_train = np.loadtxt('../trained_models/NN95_stds_train.csv',delimiter=',')
bias_train = np.loadtxt('../trained_models/NN95_bias_train.csv',delimiter=',')
bias_test = np.loadtxt('../trained_models/NN95_bias_test.csv',delimiter=',')

plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
f,axs = plt.subplots(2,1,sharey=True,sharex=True,figsize=(3.31,6.97))
one = axs[0].plot(10**(loglams_test[:len(err_train)]),best_er_train,c=(0.25,0.25,0.25),linestyle='solid')
hundred = axs[0].plot(10**(loglams_test[:len(err_train)]),err_train,c='mediumvioletred',linestyle='solid')
one_std = axs[0].plot(10**(loglams_test[:len(err_train)]),best_std_train,c=(0.25,0.25,0.25),linestyle='--')
hundred_std = axs[0].plot(10**(loglams_test[:len(err_train)]),std_train,c='mediumvioletred',linestyle='--')
#axs[0].plot(10**(loglams_test[:len(bias_train)]),bias_train,c=(0.25,0.25,0.25),linestyle='dotted')
#axs[0].plot(10**(loglams_test[:len(b_err_train)]),b_err_train,c='mediumvioletred',linestyle='dotted')
axs[1].plot(10**(loglams_test[:len(err_test)]),best_er_test,c=(0.25,0.25,0.25),label='test set $\overline{\epsilon}$')
axs[1].plot(10**(loglams_test[:len(err_test)]),best_std_test,c=(0.25,0.25,0.25),linestyle='--',label='test set 1$\sigma$')
axs[1].plot(10**(loglams_test[:len(err_test)]),err_test,c='mediumvioletred')
axs[1].plot(10**(loglams_test[:len(err_test)]),std_test,c='mediumvioletred',linestyle='--')
#axs[1].plot(10**(loglams_test[:len(bias_test)]),bias_test,c=(0.25,0.25,0.25),linestyle='dotted')
#axs[1].plot(10**(loglams_test[:len(b_err_test)]),b_err_test,c='mediumvioletred',linestyle='dotted')
#axs[0].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
axs[0].set_ylabel(r'${\rm relative\ prediction\ error}$')
axs[1].set_ylabel(r'${\rm relative\ prediction\ error}$')
plt.subplots_adjust(wspace=0, hspace=0.00)
axs[0].legend([('solid','solid'),('--','--')],['train set $\overline{\epsilon}$','train set $\sigma_{\epsilon}$'],handler_map={tuple:AnyObjectHandler()})
axs[1].legend([('solid','solid'),('--','--')],['test set $\overline{\epsilon}$','test set $\sigma_{\epsilon}$'],handler_map={tuple:AnyObjectHandler()})
#axs[0].legend([('solid','solid'),('--','--'),('dotted','dotted')],[r'${\rm train\ set\ }$' r'$\overline{\epsilon}$',r'${\rm train\ set\ }$' r'$\sigma_{\epsilon}$',r'${\rm train\ set\ bias}$'],handler_map={tuple:AnyObjectHandler()},frameon=False)
#axs[1].legend([('solid','solid'),('--','--'),('dotted','dotted')],[r'${\rm test\ set\ }$' r'$\overline{\epsilon}$',r'${\rm test\ set\ }$' r'$\sigma_{\epsilon}$',r'${\rm test\ set\ bias}$'],handler_map={tuple:AnyObjectHandler()},frameon=False)
f.legend([one,hundred],labels=[r'${\rm best\ performing\ NN}$',r'${\rm committee\ of\ 100\ NNs}$'],loc='upper center',bbox_to_anchor=(0.19,0.35,0.25,0.25),frameon=False,borderpad=0.0,borderaxespad=0.0,mode='expand')
plt.savefig('../paper/best_to_hundred.png',bbox_inches='tight',dpi=600)
plt.savefig('../paper/best_to_hundred.pdf',bbox_inches='tight',dpi=600)
plt.show()


plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
plt.figure(figsize=(3.31,3.31))
plt.plot(10**(loglams_test[:len(err_train)]),err_train/best_er_train,c=(0.25,0.25,0.25),linestyle='solid',label=r'${\rm train\ set\ }$' r'$\overline{\epsilon}$')
plt.plot(10**(loglams_test[:len(err_train)]),std_train/best_std_train,c=(0.25,0.25,0.25),linestyle='--',label=r'${\rm train\ set\ }$' r'$\sigma_{\epsilon}$')
#plt.plot(10**(loglams_test[:len(bias_train)]),bias_train/b_err_train,c=(0.25,0.25,0.25),linestyle='dotted',label=r'${\rm train\ set\ bias}$')
plt.plot(10**(loglams_test[:len(err_train)]),err_test/best_er_test,c='mediumvioletred',linestyle='solid',label=r'${\rm test\ set\ }$' r'$\overline{\epsilon}$')
plt.plot(10**(loglams_test[:len(err_train)]),std_test/best_std_test,c='mediumvioletred',linestyle='--',label=r'${\rm test\ set\ }$' r'$\sigma_{\epsilon}$')
#plt.plot(10**(loglams_test[:len(b_err_train)]),bias_test/b_err_test,c='mediumvioletred',linestyle='dotted',label=r'${\rm test\ set\ bias}$')
plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
plt.ylabel(r'${\rm relative\ prediction\ error}$')
plt.legend(frameon=False)
plt.savefig('../paper/best_to_hundred_comp_bias_ratio.png',bbox_inches='tight',dpi=600)
plt.savefig('../paper/best_to_hundred_comp_bias_ratio.pdf',bbox_inches='tight',dpi=600)
plt.show()


best_er_test = np.loadtxt('../Davies/Davies_test_epsilon_bar.txt')
best_std_test = np.loadtxt('../Davies/Davies_test_epsilon_sigma.txt')
best_er_train = np.loadtxt('../Davies/Davies_train_epsilon_bar.txt')
best_std_train = np.loadtxt('../Davies/Davies_train_epsilon_sigma.txt')
bias_train = np.loadtxt('../Davies/Davies_train_bias.txt')
bias_test = np.loadtxt('../Davies/Davies_test_bias.txt')

plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
f,axs = plt.subplots(2,1,sharey=True,sharex=True,figsize=(3.31,6.97))
one = axs[0].plot(10**(loglams_test[:len(err_train)]),best_er_train,c=(0.25,0.25,0.25),linestyle='solid')
hundred = axs[0].plot(10**(loglams_test[:len(err_train)]),err_train,c='mediumvioletred',linestyle='solid')
one_std = axs[0].plot(10**(loglams_test[:len(err_train)]),best_std_train,c=(0.25,0.25,0.25),linestyle='--')
hundred_std = axs[0].plot(10**(loglams_test[:len(err_train)]),std_train,c='mediumvioletred',linestyle='--')
#axs[0].plot(10**(loglams_test[:len(bias_train)]),bias_train,c=(0.25,0.25,0.25),linestyle='dotted')
#axs[0].plot(10**(loglams_test[:len(b_err_train)]),b_err_train,c='mediumvioletred',linestyle='dotted')
axs[1].plot(10**(loglams_test[:len(err_test)]),best_er_test,c=(0.25,0.25,0.25),label='test set $\overline{\epsilon}$')
axs[1].plot(10**(loglams_test[:len(err_test)]),best_std_test,c=(0.25,0.25,0.25),linestyle='--',label='test set 1$\sigma$')
axs[1].plot(10**(loglams_test[:len(err_test)]),err_test,c='mediumvioletred')
axs[1].plot(10**(loglams_test[:len(err_test)]),std_test,c='mediumvioletred',linestyle='--')
#axs[1].plot(10**(loglams_test[:len(bias_test)]),bias_test,c=(0.25,0.25,0.25),linestyle='dotted')
#axs[1].plot(10**(loglams_test[:len(b_err_test)]),b_err_test,c='mediumvioletred',linestyle='dotted')
#axs[0].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
axs[1].set_xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
axs[0].set_ylabel(r'${\rm relative\ prediction\ error}$')
axs[1].set_ylabel(r'${\rm relative\ prediction\ error}$')
plt.subplots_adjust(wspace=0, hspace=0.00)
#axs[0].legend([('solid','solid'),('--','--')],['train set $\overline{\epsilon}$','train set $\sigma_{\epsilon}$'],handler_map={tuple:AnyObjectHandler()})
#axs[1].legend([('solid','solid'),('--','--')],['test set $\overline{\epsilon}$','test set $\sigma_{\epsilon}$'],handler_map={tuple:AnyObjectHandler()})
axs[0].legend([('solid','solid'),('--','--'),('dotted','dotted')],[r'${\rm train\ set\ }$' r'$\overline{\epsilon}$',r'${\rm train\ set\ }$' r'$\sigma_{\epsilon}$',r'${\rm train\ set\ bias}$'],handler_map={tuple:AnyObjectHandler()},frameon=False)
axs[1].legend([('solid','solid'),('--','--'),('dotted','dotted')],[r'${\rm test\ set\ }$' r'$\overline{\epsilon}$',r'${\rm test\ set\ }$' r'$\sigma_{\epsilon}$',r'${\rm test\ set\ bias}$'],handler_map={tuple:AnyObjectHandler()},frameon=False)
f.legend([one,hundred],labels=[r'${\rm Davies\ et\ al.\ (2018)}$',r'${\rm QSANNdRA}$'],loc='upper center',bbox_to_anchor=(0.19,0.35,0.25,0.25),frameon=False,borderpad=0.0,borderaxespad=0.0,mode='expand')
plt.savefig('../paper/Davies_comp_nobias.png',bbox_inches='tight',dpi=600)
plt.savefig('../paper/Davies_comp_nobias.pdf',bbox_inches='tight',dpi=600)
plt.show()

#dat = np.loadtxt('../NN/error_test_committee.csv',delimiter=',')
#err_test = dat[:,0]
#std_test = dat[:,1]
plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
plt.figure(figsize=(3.31,3.31))
plt.plot(10**(loglams_test[:len(err_train)]),err_train/best_er_train,c=(0.25,0.25,0.25),label=r'${\rm train\ set\ }$' r'$\overline{\epsilon}$' r'${\rm \ ratio}$')
plt.plot(10**(loglams_test[:len(err_train)]),std_train/best_std_train,c=(0.25,0.25,0.25),label=r'${\rm train\ set\ }$' r'$\sigma_{\epsilon}$' r'${\rm \ ratio}$',linestyle='--')
#plt.plot(10**(loglams_test[:len(b_err_train)]),b_err_train/bias_train[:-1],c=(0.25,0.25,0.25),linestyle='dotted',label=r'${\rm train\ set\ bias}$')
plt.plot(10**(loglams_test[:len(err_test)]),err_test/best_er_test,c='mediumvioletred',label=r'${\rm test\ set\ }$' r'$\overline{\epsilon}$' r'${\rm \ ratio}$')
plt.plot(10**(loglams_test[:len(err_test)]),std_test/best_std_test,c='mediumvioletred',label=r'${\rm test\ set\ }$' r'$\sigma_{\epsilon}$' r'${\rm \ ratio}$',linestyle='--')
#plt.plot(10**(loglams_test[:len(b_err_test)]),b_err_test/bias_test[:-1],c='mediumvioletred',linestyle='dotted',label=r'${\rm test\ set\ bias}$')
plt.xlabel(r'${\rm rest\ wavelength\ [}$' r'$\mathrm{\AA}$' r'${\rm ]}$')
plt.ylabel(r'${\rm relative\ prediction\ error}$')
plt.legend(frameon=False)
plt.savefig('../paper/Davies_ratio_7.0_train.png',bbox_inches='tight',dpi=600)
plt.savefig('../paper/Davies_ratio_7.0_train.pdf',bbox_inches='tight',dpi=600)
plt.show()


--------------------------------

for i in range(0,len(proj_pred_b)-1):
	if np.max(proj_pred_b[i,:])<2:
		print(i)
		plt.plot(10**(loglams_test[:len(err_train)]),proj_pred_b[i,:])

m=723
n = m + len(train_red_orig)
hdu_raw = pf.open('../QSO_spectra_7/'+str(IDs.iloc[n,0]))
hdu = hdu_raw[1].data #contains the data after primary smoothing
#spec_id = [str(IDs.iloc[len(train)+n,0])] #row label
loglam = hdu['loglam'] - np.log10(1+hdu_raw[2].data['Z']) #wavelengths calibrated for redshifts
flux = np.divide(hdu['flux'],IDs.iloc[n,1])
noise = np.divide(np.sqrt(1/hdu['ivar']),IDs.iloc[n,1])
hdu = []
hdu_raw.close()
plt.rc('text',usetex=True)
font = {'family':'sans-serif','sans-serif':['Helvetica'],'size':8}
plt.rc('font',**font)
plt.figure(figsize=(3.31,3.31))
plt.xlim([1180,1350])
plt.ylim([-0.02,2.5])
id = str(IDs.iloc[n,0])[:-5]
plt.plot(10**loglam,flux,c=(0.25,0.25,0.25),label=id)				
plt.plot(10**(loglams_test),test.values[m,:],c='c')
plt.plot(10**(loglams_test[:len(err_train)]),predictions_test[m,:,:],c='m',linewidth = 1,alpha=0.05)
plt.plot(10**(loglams_test[:len(err_train)]),proj_pred_b[m,:],c='mediumvioletred',linewidth=3)
plt.legend(frameon=False)
plt.savefig('../prediction_'+id+'.png',bbox_inches='tight',dpi=600)
plt.show()

35 - 'spec-4040-55605-0102'
98
114
122 - 'spec-4097-55506-0536'
129
131
178
228 - 'spec-4314-55855-0704'
310
313
317
348
418
451
477
594
597
723
749
804
821
947
951
981
1076
1132
1235
1273
1277
1285
1331
1349
1399
1453
1494
1504
1563
1670
1696
1727
1769
1799
1830
1851
2039
2055
2069
2108
2110
2184
2192
2288
2292
2306
2307
2428
2442
2473
2479
2569
2712
2724
