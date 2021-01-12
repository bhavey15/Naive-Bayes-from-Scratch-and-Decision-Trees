from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
from sklearn.linear_model import LinearRegression
from copy import deepcopy


def bootstraping(X_train,X_test,y_train,y_test):
'''Performs bootstraping sampling from the train data,
    trains Linear Regression the models on the samples 
    and predict the results for test data
    Inputs = X_train, X_test,y_train, y_test
    Output = MSE - Bias**2 - Variance'''
  n_samples=500
  y_pred=np.zeros(y_test.shape)
  #y_values stores the predicted values for test data for all bootstrap samples generated
  y_values=np.ndarray((len(X_test),n_samples))
  for i in range(n_samples):
    pop=np.random.choice(np.random.randint(0,len(X_train),len(X_train)),len(X_train))
    X_train_1=X_train[pop].reshape(-1,1)
    y_train_1=y_train[pop]
    model=LinearRegression().fit(X_train_1,y_train_1)
    y_pred=y_pred+model.predict(X_test)
    y_values[:,i]=model.predict(X_test)

  #y_pred stores the mean

  y_pred=y_pred/n_samples
  y_pred=y_pred.reshape(-1,1)
  bias=np.subtract(y_pred.ravel(),y_test)
  variance=((y_values-y_pred)**2).sum(axis=1)/(n_samples-1)
  mag_variance=variance.T@variance
  mag_bias=bias.T@bias
  mse=((y_values-y_test.reshape(-1,1))**2).sum(axis=1)/n_samples
  mag_mse=mse.T@mse
  final_val=mse-bias**2-variance
  return (final_val.T@final_val)**(1/2)



if __name__ == '__main__':
  df3=pd.read_csv('/content/drive/My Drive/ML DataSet/weight-height.csv')
  df3=df3.replace({'Male':0, 'Female':1})
  X=df3['Height'].to_numpy()
  y=df3['Weight'].to_numpy()
  X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=42)
  X_test=X_test.reshape(-1,1)
  bootstraping(X_train,X_test,y_train,y_test)
