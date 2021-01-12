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
from sklearn.tree import DecisionTreeClassifier
from copy import deepcopy

def ret_indices(fold_size,indices,k):
  '''returns indices of val and train sets'''
  val_indices = indices[((k-1)*fold_size):(k*fold_size)]
  train_indices = np.setdiff1d(indices, val_indices)
  val_indices = np.array(val_indices)
  return (val_indices, train_indices)
def k_fold_val(X,y,k,model):
  '''does the K-fold validation and returns the val accuracy for each fold'''
  accuracies=[]
  indices=[i for i in range(len(y))]
  np.random.shuffle(indices)
  fold_size=int(len(X)/k)
  accuracy=0.0
  for i in range(k):
    val_indices,train_indices=ret_indices(fold_size,indices,k)
    X_train=X.iloc[train_indices]
    y_train=y.iloc[train_indices]
    X_val=X.iloc[val_indices]
    y_val=y.iloc[val_indices]
    model.fit(X_train,y_train)
    y_pred=model.predict(X_val)
    accuracies.append(accuracy_score(y_pred,y_val))
  return np.array(accuracies)

def choose_depth(X,y,tree_depths,k):
  '''returns the mean and std deviation of the validation accuracies for every depth in the list. Also returns a list of train accuracies'''
  mean=[]
  std=[]
  acc_score=[]
  for depth in tree_depths:
    model=DecisionTreeClassifier(max_depth=depth)
    accuracies_cv=k_fold_val(X,y,k,model)
    mean.append(accuracies_cv.mean())
    std.append(accuracies_cv.std())
    acc_score.append(model.fit(X,y).score(X,y))
  mean=np.array(mean)
  std=np.array(std)
  acc_score=np.array(acc_score)
  return mean,std,acc_score

def plot_depth_cv(tree_depths,mean,std,acc_score):
  '''plots the validation and train accuracies wrt the tree depth'''
  fig, ax = plt.subplots(1,1, figsize=(15,5))
  ax.plot(tree_depths, mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
  ax.fill_between(tree_depths, mean-2*std, mean+2*std, alpha=0.2)
  # ylim = plt.ylim()
  ax.plot(tree_depths, acc_score, '-*', label='train accuracy', alpha=0.9)
  ax.set_title('accuracy vs depth', fontsize=16)
  ax.set_xlabel('Tree depth', fontsize=14)
  ax.set_ylabel('Accuracy', fontsize=14)
  # ax.set_ylim(ylim)
  ax.set_xticks(tree_depths)
  ax.legend()
def evaluation_metrics(y_pred,y,proba,type):
  '''Input: y_pred = predicted values from a model, y = the actual values
    proba= the probabilities of all samples in each of the classes as returned by the model
    type= Binary or Multiclass

    Output: 
    Binary = '''
  y=y.to_numpy().ravel()
  classes=np.unique(y)
  confusion_matrix=np.zeros((len(classes),len(classes)))
  for i,j in zip(y_pred,y):
      confusion_matrix[int(i),int(j)]+=1
  sns.heatmap(confusion_matrix)
  cf={}
  if(type==1):
    for i in classes:
      ccf={}
      tp=len(y[np.logical_and(y_pred==i,y==i)])
      tn=len(y[np.logical_and(y_pred!=i,y!=i)])
      fp=len(y[np.logical_and(y_pred==i,y!=i)])
      fn=len(y[np.logical_and(y_pred!=i,y==i)])
      ccf['tp']=tp
      ccf['tn']=tn
      ccf['fn']=fn
      ccf['fp']=fp
      cf[i]=ccf
    macro_precision=0
    macro_recall=0
    micro_precision_num=0
    micro_precision_den=0
    micro_recall_num=0
    micro_recall_den=0
    for i in classes:
      macro_precision+=(cf[i]['tp']/(cf[i]['tp']+cf[i]['fp']))
      micro_precision_num+=cf[i]['tp']
      micro_precision_den+=cf[i]['tp']+cf[i]['fp']
      macro_recall+=(cf[i]['tp']/(cf[i]['tp']+cf[i]['fn']))
      micro_recall_num+=cf[i]['tp']
      micro_recall_den+=cf[i]['tp']+cf[i]['fn']
    micro_precision=micro_precision_num/micro_precision_den
    macro_precision=macro_precision/len(classes)
    macro_recall=macro_recall/len(classes)
    micro_recall=micro_recall_num/micro_recall_den
    macro_f1=2*macro_recall*macro_precision/(macro_recall+macro_precision)
    micro_f1=2*micro_recall*micro_precision/(micro_recall+micro_precision)
    return micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1,confusion_matrix
  else:
    tp=len(y[np.logical_and(y_pred==y,y==1)])
    tn=len(y[np.logical_and(y_pred==y,y==0)])
    fp=len(y[np.logical_and(y_pred!=y,y==0)])
    fn=len(y[np.logical_and(y_pred!=y,y==1)])
    cf['tp']=tp
    cf['tn']=tn
    cf['fp']=fp
    cf['fn']=fn
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*recall*precision/(recall+precision)
    accuracy=(tp+tn)/(tp+fn+tn+fp)
    threshold=np.linspace(0,1.1,40)
    tpr_list=np.zeros(len(threshold))
    fpr_list=np.zeros(len(threshold))
    ind=0
    for t in threshold:
      y_p=np.zeros(y.shape[0])
      prob1=proba[:,1]
      y_p[prob1>=t]=1
      tp=len(y[np.logical_and(y_p==y,y==1)])
      tn=len(y[np.logical_and(y_p==y,y==0)])
      fp=len(y[np.logical_and(y_p!=y,y==0)])
      fn=len(y[np.logical_and(y_p!=y,y==1)])
      tpr=tp/(tp+fn)
      fpr=fp/(fp+tn)
      tpr_list[ind]=tpr
      fpr_list[ind]=fpr
      ind+=1
    
    plt.plot(fpr_list,tpr_list)
    plt.xlabel("False Positives Rate")
    plt.ylabel("True positives rate")
    return recall,f1,precision,accuracy,confusion_matrix
def bonus_ROC_curve(y_pred,y,prob):
  '''plots ROC curve for multiclass classification'''
  '''for individual classes and macro and micro ROC curves'''
  y=y.to_numpy().ravel()
  classes=np.unique(y)
  threshold=np.linspace(0,1.05,100)
  matpr_list=np.zeros(len(threshold))
  mafpr_list=np.zeros(len(threshold))
  mitpr_list=np.zeros(len(threshold))
  mifpr_list=np.zeros(len(threshold))
  ind=0
  tpr_list=np.zeros((len(threshold),len(classes)))
  fpr_list=np.zeros((len(threshold),len(classes)))
  for t in threshold:
    matpr=0
    mafpr=0
    mifpr_num=0
    mifpr_den=0
    mitpr_num=0
    mitpr_den=0
    for i in classes:
      y_p=np.zeros(y.shape[0])
      prob1=prob[:,int(i)]
      if(i==0):
        y_p[prob1<t]=1
      else:
        y_p[prob1>=t]=i
      tp=len(y[np.logical_and(y_p==y,y==i)])
      tn=len(y[np.logical_and(y_p==y,y!=i)])
      fp=len(y[np.logical_and(y_p!=y,y!=i)])
      fn=len(y[np.logical_and(y_p!=y,y==i)])
      matpr+=tp/(tp+fn)
      mafpr+=fp/(fp+tn)
      mitpr_num+=tp
      mitpr_den+=tp+fn
      mifpr_num+=fp
      mifpr_den+=fp+tn
      tpr_list[ind,int(i)]=matpr
      fpr_list[ind,int(i)]=mafpr
    matpr_list[ind]=matpr/len(classes)
    mafpr_list[ind]=mafpr/len(classes)
    mitpr_list[ind]=mitpr_num/mitpr_den
    mifpr_list[ind]=mifpr_num/mifpr_den
    ind+=1
  for i in classes:
    i=int(i)
    plt.plot(fpr_list[:,i],tpr_list[:,i],label=i)
  plt.plot(mafpr_list,matpr_list,label="macro")
  plt.plot(mifpr_list,mitpr_list,label="micro")
  plt.xlabel("False positive rate")
  plt.ylabel("true positive rate")
  plt.legend()
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

# def savemodel(model):
#   filename='bestmodel.sav'
#   pickle.dump(model,open(filename,'wb'))
# def loadmodel():
#   filename='bestmodel.sav'
#   return pickle.load(open(filename,'rb'))




if __name__ == '__main__':
  X,y=generate_dataset('A')
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
  X_train, X_val, y_train, y_val=train_test_split(X_train,y_train,test_size=0.2)
  
  clf=DecisionTreeClassifier()
  # clf=GaussianNB()
  clf.fit(X_train,y_train)
  y_pred=clf.predict(X_test)
  print(accuracy_score(y_pred,y_test))
  y_predict_proba=clf.predict_proba(X_test)
  tree_depths=range(1,150)
  mean,std,acc_score=choose_depth(X_train,y_train,tree_depths,5)
  max_depth= 1+np.argmax(mean)
  plot_depth_cv(tree_depths,mean,std,acc_score)
  print(np.max(mean),max_depth)  #print accuracy and max depth

  #save the best model
  filename='bestmodel.sav'
  model=DecisionTreeClassifier(max_depth=max_depth)
  model.fit(X_train,y_train)
  pickle.dump(model,open(filename,'wb'))
  
  #load the best model
  model_pickle=pickle.load(open(filename,'rb'))
  y_test=y.predict(X_test)
  y_predict_proba=model.predict_proba(X_test)
  #for binary classification
  print(evaluation_metrics(y_pred,y_test,y_predict_proba,2))
  #for multiclass classification
  print(evaluation_metrics(y_pred,y_test,y_predict_proba,1))
  bonus_ROC_curve(y_pred,y_test,y_predict_proba)