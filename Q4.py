from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from copy import deepcopy

class Gaussian_NB:

  	def __init__(self):
    	self.mean_variance={}
    	self.prior={}
  	def calc_prior(self,y):
    	self.prior=(y.value_counts()/len(y)).to_dict()
  	def calc_probability(self,x,mean,var):
    	return (1 / (math.sqrt(2 * math.pi * var)))*(math.exp(-(math.pow(x - mean, 2) / (2 * var))))
  	def calc_mean_variance(self,X,y):
    	self.mean_variance={}
    	for c in np.unique(y.to_numpy()):
      	X_new=X[y==c]
      	mv={}
      	for feat in range(len(X.T)):
        	mv[feat]=[]
        	mv[feat].append(X_new[feat].mean(axis=0))
        	mv[feat].append(X_new[feat].std(axis=0)**2)
      	self.mean_variance[c]=mv
  	def fit(self,X,y):
    	self.calc_prior(y)
    	self.calc_mean_variance(X,y)
  	def predict(self,X):
    	y_pred=np.zeros((len(X),1))
    	for i in range(len(X)):
      	results={}
     	for k,v in self.prior.items():
        	p=0
        	for feat in range(len(X.T)):
          	prob=self.calc_probability(X.iloc[i,feat],self.mean_variance[k][feat][0],max(10**-6,self.mean_variance[k][feat][1]))
          	if prob>0:
            	p=p+math.log(prob)
            	# p=p*prob
        	results[k]=math.log(v)+p
      	y_pred[i]=np.argmax(results)
    	return y_pred
def generate_dataset(type):
	if(type=='A'):
		df=h5py.File('/content/drive/My Drive/ML DataSet/part_A_train.h5')
		X=pd.DataFrame(df.get('X'))
		y=pd.DataFrame(df.get('Y'))
		y=0*y[0]+1*y[1]+2*y[2]+3*y[3]+4*y[4]+5*y[5]+6*y[6]+7*y[7]+8*y[8]+9*y[9]
		y=pd.DataFrame(y)
		return X,y
	if(type=='B'):
		df=h5py.File('/content/drive/My Drive/ML DataSet/part_B_train.h5')
		X=pd.DataFrame(df.get('X'))
		y=pd.DataFrame(df.get('Y'))
		y=0*y[0]+1*y[1]
		y=pd.DataFrame(y)
		return X,y


if __name__ == '__main__':
	X,y=generate_dataset('A')
	obj=Gaussian_NB()
	y=y.astype(int)
	obj.fit(X,y)
	y_pred=obj.predict(X)
	print(accuracy_score(y_pred,y))
	clf=GaussianNB()
	clf.fit(X,y)
	y_pred_1=clf.predict(X)
	accuracy_score(y_pred_1,y.to_numpy())

