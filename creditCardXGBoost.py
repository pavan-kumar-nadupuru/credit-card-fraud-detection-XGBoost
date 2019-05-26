#!/usr/bin/env python
# coding: utf-8

#Befre running the code download the dataset from 'https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/3' and extract it
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import f1_score

try:
    dataframe=pd.read_csv('creditcard.csv')
except:
    print('------------------CAN NOT RUN---------------------------------I/O ERROR-----------------------------FILE TO LOAD IS NOT FOUND-----------DOWNLOAD')
    print("Befre running the code download the dataset from 'https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/3' and extract it")
    print('-------------------------------------------------------------------------------------------------------------------------------------------------')

print ("Dimensions of dataset are:", dataframe.shape)
print ("Normal transaction : ", dataframe['Class'][dataframe['Class'] == 0].count())
print ("Fraudulent transaction:", dataframe['Class'][dataframe['Class'] == 1].count())

class0 = dataframe.query('Class == 0')
class1 = dataframe.query('Class == 1')
class0 = class0.sample(frac=1)
class1 = class1.sample(frac=1)

# ALLclass0train = class0
# ALLclass1train = class1
# ALLtrain = ALLclass0train.append(ALLclass1train, ignore_index=True).values
# ALL_X=ALLtrain[:,0:30]
# ALL_Y=ALLtrain[:,30]

class0train = class0.iloc[0:5508]
class1train = class1
train = class0train.append(class1train, ignore_index=True).values


X = train[:,0:30]
Y = train[:,30]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=73)

model = XGBClassifier()
model.fit(X_train,Y_train)

model.score(X_test,Y_test)
Y_pred=model.predict(X_test)

#Y_forALL=model.predict(X)

f1=f1_score(Y_test,Y_pred)
print(f'The model generalizes to new data that it has not seen to an accuracy of {model.score(X_test,Y_test)*100}% and an F1 score of {(round(f1*100))/100} over test set ')

try:
    joblib.dump(model,"CCFD.pkl")
    print('Saved model to CCFD.pkl file for further predictions')
except:
    print('Could not save model')
