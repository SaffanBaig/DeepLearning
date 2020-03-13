import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencode_X_1 = LabelEncoder()
labelencode_X_2 = LabelEncoder()

X[:, 1] = labelencode_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencode_X_1.fit_transform(X[:, 2])
onehotencode = OneHotEncoder(categorical_features=[1])
X = onehotencode.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()   
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer="uniform", activation = 'relu', input_dim = 11))
    classifier.add(Dense(6, kernel_initializer="uniform", activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer="uniform", activation = 'sigmoid'))    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
mean = accuracies.mean()
variance = accuracies.std()




