# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 00:57:24 2023

@author: tarun
"""

## ANN practice

import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__


dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values #taken out first 3 columns since they are irrelevant to the process. Kept all row. This is a two dimensional matrix
y = dataset.iloc[:, -1].values #selected only the last column since it is the dependent variable. It is a single dim matrix

print(X)
print(y)


## now encode categotical data, country and gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

##now print x and make sure that it is coded. In the case of gender we have used label conding since male and female has a relationship. Meaning if it is not female, it has to be male and vice and versa
print(X[:,2])


##now encode country - we will use one hot encoding since it is a categorical data and has no order or relationship. It does not mean that if it in not france then it has to be spain or vice versa. 
#So no relationship so we will use one hot encoding that we create seperate column for each country to represent as #
#1 and 0. This way data is handled best when there is no relationship

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X= np.array(ct.fit_transform(X))
print(X)


##now split the dataset into training set and test set using scikit learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


### Feature Scaling - It is very very important when it comes to deep learning. it is very important. WHY ask chtgpt
##we will standardize all data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



