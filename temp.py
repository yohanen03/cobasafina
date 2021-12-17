# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data = pd.read_csv('diabetes_data_upload1.csv',sep=";")

Gender=pd.get_dummies(data['Gender'],drop_first=True)
Polyuria=pd.get_dummies(data['Polyuria'],drop_first=True)
Polydipsia=pd.get_dummies(data['Polydipsia'],drop_first=True)
suddenweightloss=pd.get_dummies(data['suddenweightloss'],drop_first=True)
weakness=pd.get_dummies(data['weakness'],drop_first=True)
Polyphagia=pd.get_dummies(data['Polyphagia'],drop_first=True)
Genitalthrush=pd.get_dummies(data['Genitalthrush'],drop_first=True)
visualblurring=pd.get_dummies(data['visualblurring'],drop_first=True)
Itching=pd.get_dummies(data['Itching'],drop_first=True)
Irritability=pd.get_dummies(data['Irritability'],drop_first=True)
delayedhealing=pd.get_dummies(data['delayedhealing'],drop_first=True)
partialparesis=pd.get_dummies(data['partialparesis'],drop_first=True)
musclestiffness=pd.get_dummies(data['musclestiffness'],drop_first=True)
Alopecia=pd.get_dummies(data['Alopecia'],drop_first=True)
Obesity=pd.get_dummies(data['Obesity'],drop_first=True)
classes=pd.get_dummies(data['classes'],drop_first=True)

data.drop(['Gender','Polyuria','Polydipsia','suddenweightloss','weakness','Polyphagia','Genitalthrush','visualblurring','Itching','Irritability','delayedhealing','partialparesis','musclestiffness','Alopecia','Obesity','classes'],axis=1,inplace=True)

data = pd.concat([Gender,Polyuria,Polydipsia,suddenweightloss,weakness,Polyphagia,Genitalthrush,visualblurring,Itching,Irritability,delayedhealing,partialparesis,musclestiffness,Alopecia,Obesity,classes],axis=1)

X=data.iloc[:,0:15].values
y=data.iloc[:,-1].values

#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report

# Import libarary confusion matrix
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.model_selection import train_test_split

# Import libarary Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

logreg = LogisticRegression(solver= 'lbfgs',max_iter=400)
logreg.fit(X_train, y_train)

pickle.dump(logreg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
