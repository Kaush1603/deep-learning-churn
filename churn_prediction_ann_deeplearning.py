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


##Building ANN

ann = tf.keras.models.Sequential() ##The Sequential function helps you define the structure of this network by 
                                   #providing a way to add and arrange the layers
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))   ###now we are using add method from object of sequential 
                                                            #to add input and hidden layer. Input layer will have all the inputs from the data but for hidden layer we have to choose unit. 
                                                            #There is no rule to choose how many unit or neuron, it is on trial basis so it is one of the hyperparameter. Mostly activation fuction that is used  in hidden layer is rectifier fuction
                                                            #and code for that is "relu". It is shallow neural network since it there is only one hidden layer. So it is not deep enough lol
###now adding another layer so that this is deep learning and we will simply copy paste the above code and add method will add another layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

##now adding output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) ##now for output layer there will be changes in hyperparamter such a units will be changed, in this case the output is only binary so dimen is 1 so only one neuron is required
                                                           #but if it were to be a categorical data with 3 option then one hot coding would have been required and then would haver required 3 neuron, output example would be (0,0,1) and activation for output, we would like to have sigmoid activation fuction -
                                                           #why? it gives only the prediction but also the probability of prediction. So it means that if it is 1 meaning customer will leave but also the probablity that it will be leave
## Compiling the ANN
ann.compile(optimizer= 'adam' , loss='binary_crossentropy' , metrics= ['accuracy']) ## best to use optimizer will be that can use stochastic gradiaent descent and it is adam optimizer, there are others as well. Loss fuction will calculate the difference between predicted and actual number. Remember we want it in parabolic or unidirectional
## whenever we are doing binary classification then loss fuction should always be "binary_crossentropy'. If it were categorical outputs then loss fuction would phave been categorical_crossentropy. Also be non binary then activation fuction should be softmax in output layer.

###Training the ANN on the Training set

ann.fit(X_train, y_train, batch_size=32, epochs=100) ##in ANN batch is the preferred method. Instead of giving value one by one, we will give a batch of values which it will predict and compare it with real values

## now we will predict with given inputs

##when using predict always use [[]] bracket since predict method always expect 2d array
##given inputs are Geography: France, Credit Score: 600, Gender: male, Age: 40 years old, Tenure: 3 years, balance: $6000
##number of product 2 and so on,,,
##geography will be entered as one hot coding code

print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))) ##we should be calling this method on the same scaling on which the training was applied
##also we will not use fit_transform as it will again tranform and there is be data leaked so we have to use tranform method in order to keep it consistent with the old scaling
##output is 0.0216 so the probablity is very low

###now predict it with test set results

y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) #comparing the result side by side

### confusion matrix for accuracy

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
