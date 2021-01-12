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
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
def stratified_sampling(X,y):
  sss=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
  sss.get_n_splits(X,y)
  X=np.array(X)
  y=np.array(y)
  for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
  #counting class frequencies
  # train_values,train_count=np.unique(y_train,return_counts=True)
  # test_values,test_count=np.unique(y_test,return_counts=True)
  # total_values, total_count=np.unique(y,return_counts=True)
  # print(train_values,train_count,'test_count',test_count,'total_count',total_count)
  #class frequencies of each class is divided in the same ratio as the train/test data(80:20)
  X_train=pd.DataFrame(X_train)
  X_test=pd.DataFrame(X_test)
  y_train=pd.DataFrame(y_train)
  y_test=pd.DataFrame(y_test)
  return X_train,X_test,y_train,y_test

def PCA_decomposition(X,y):
  pca=PCA(n_components=50)
  pca_res=pca.fit_transform(X)
  X['pca-one'] = pca_res[:,0]
  X['pca-two'] = pca_res[:,1] 
  perm=np.random.permutation(X.shape[0])
  plt.figure(figsize=(8,8))
  sns.scatterplot(
      x="pca-one", y="pca-two",
      hue=y.to_numpy().ravel()[perm],
      palette=sns.color_palette("hls", 10),
      data=X.loc[perm,:],
      legend="full",
      alpha=0.9
  )
  return pca_res,y

def TSNE_analysis(X,y):
  tsne=TSNE(n_components=2)
  X_tsne=tsne.fit_transform(X)
  perm=np.random.permutation(X.shape[0])
  X['tsne_one']=X_tsne[perm,0]
  X['tsne_two']=X_tsne[perm,1]
  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x="tsne_one", y="tsne_two",
      hue=y.to_numpy().ravel()[perm],
      palette=sns.color_palette("hls", 10),
      data=X,
      legend="full",
      alpha=1
  )

def SVD_decomposition(X,y):
  svd=TruncatedSVD(n_components=50)
  X_svd=svd.fit_transform(X)
  X['svd-one'] = X_svd[:,0]
  X['svd-two'] = X_svd[:,1] 
  X['svd-three'] = X_svd[:,2]
  perm=np.random.permutation(X.shape[0])
  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x="svd-one", y="svd-two",
      hue=y.to_numpy().ravel()[perm],
      palette=sns.color_palette("hls", 10),
      data=X.loc[perm,:],
      legend="full",
      alpha=1
  )
def generate_dataset(type):
  if(type=='A'):
    df=h5py.File('/content/drive/My Drive/ML DataSet/part_A_train.h5')
    X=pd.DataFrame(df.get('X'))
    y=pd.DataFrame(df.get('Y'))
    y=0*y[0]+1*y[1]+2*y[2]+3*y[3]+4*y[4]+5*y[5]+6*y[6]+7*y[7]+8*y[8]+9*y[9]
    y=pd.DataFrame(y)
    return X,y
if __name__ == '__main__':
  X,y=generate_dataset('A')
  X_pca,y_pca=PCA_decomposition(X,y)
  X_train,X_test,y_train,y_test=stratified_sampling(X_pca,y_pca)
  clf=LogisticRegression()
  clf.fit(X_train,y_train)
  from sklearn.metrics import accuracy_score
  accuracy_score(clf.predict(X_test),y_test)
  X_svd,y_svd=PCA_decomposition(X,y)
  X_train,X_test,y_train,y_test=stratified_sampling(X_svd,y_svd)
  clf=LogisticRegression()
  clf.fit(X_train,y_train)
  from sklearn.metrics import accuracy_score
  accuracy_score(clf.predict(X_test),y_test)
  TSNE_analysis(X_train,y_train)