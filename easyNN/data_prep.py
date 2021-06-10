# Copyright © Avhijit Nair and Ashutosh Singh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from keras.utils import np_utils
import seaborn as sns

def data_prep(filename,drop_features=None,train_percent=60,test_percent=20,val_percent=20,target_variable,
	pca=False,pca_data=None,pca_components=3):
	sdss_df = pd.read_csv(filename,encoding='utf-8')
	sdss_df = sdss_df.sample(frac=1)
	if drop_features is not None:
		sdss_df.drop(drop_features,axis=1)

	train_count = (train_percent/100)*sdss_df.shape[0]
	test_count = (test_percent/100)*sdss_df.shape[0]
	val_count = (val_percent/100)*sdss_df.shape[0]

	#train,validation and test split
	train_df = sdss_df.iloc[:train_count]
	validation_df = sdss_df.iloc[train_count:train_count+val_count]
	test_df = sdss_df.iloc[-test_count:]

	X_train = train_df.drop([target_variable], axis=1)
	X_validation = validation_df.drop([target_variable], axis=1)
	X_test = test_df.drop([target_variable], axis=1)

	#one-hot encoding of labels
	le = LabelEncoder()
	le.fit(sdss_df[target_variable])
	encoded_Y = le.transform(sdss_df[target_variable])
	onehot_labels = np_utils.to_categorical(encoded_Y)

	y_train = onehot_labels[:train_count]
	y_validation = onehot_labels[train_count:train_count+val_count]
	y_test = onehot_labels[-test_count:]

	#scaling of data
	scaler = StandardScaler()
	scaler.fit(X_train) # fit scaler to training data only
	X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
	X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=X_validation.columns)

	if pca:
		pc_a = PCA(n_components=pca_components)
		dfs = [X_train, X_validation, X_test]
		for i in range(len(dfs)):
			df = dfs[i]
			tranform_data = pca.fit_transform(df[pca_data])
			df = pd.concat((df, pd.DataFrame(tranform_data)), axis=1)
			df.rename({0: 'PCA1', 1: 'PCA2', 2: 'PCA3'}, axis=1, inplace=True)
			df.drop(pca_data, axis=1, inplace=True)
			dfs[i] = df
		X_train, X_validation, X_test = dfs

	return X_train,X_test,X_validation,y_train,y_test,y_validation
