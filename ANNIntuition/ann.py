# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:42:43 2020

@author: Haier
"""

import tensorflow as tf
import theano

""" Data Preprocessing """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, -1].values

#Encode the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencode_X_1 = LabelEncoder()
labelencode_X_2 = LabelEncoder()

X[:, 1] = labelencode_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencode_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting the dataset into Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()   
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dense(6, kernel_initializer="uniform", activation = 'relu', input_dim = 11))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(6, kernel_initializer="uniform", activation = 'relu'))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(1, kernel_initializer="uniform", activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

#Predicting the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Testing on single new observation
""" Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

#Tuning
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer="uniform", activation = 'relu', input_dim = 11))
    classifier.add(Dense(6, kernel_initializer="uniform", activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer="uniform", activation = 'sigmoid'))    
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25, 32], 
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10
                           )
grid_search = grid_search.fit(X_train, Y_train)
best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_score_



