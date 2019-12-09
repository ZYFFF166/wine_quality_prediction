# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:15:05 2019
The code can be run by part. The input files should be in the same directory


@author: tianh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Part 1 - data preprocessing
# Importing the dataset
dataset = pd.read_csv('redwinequality.csv')
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values

# =============================================================================
# # Encoding categorical Y data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
labelencoder_y.fit(y)
encoded_y = labelencoder_y.transform(y)

# =============================================================================

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size = 0.2, random_state = 0, stratify = encoded_y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#SMOTE

from imblearn.over_sampling import SMOTE
sm = SMOTE(ratio={0:200,1:200, 5:200}, random_state = 0)
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

#convert SMOTed data for visualization
y_train_temp = y_train + 3
y_train_smote_temp = y_train_sm + 3


# Part 2 - build the ANN architectures

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import pyplot

# Architecture 1
# Build the ANN
classifier = Sequential()
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
#keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the ANN
history = classifier.fit(X_train_sm, y_train_sm, validation_data = (X_test, y_test), batch_size = 20, epochs = 100)


# Plot accuracy during training
plt.subplot(211)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.grid(color='b', linestyle='--', linewidth=0.5)

plt.subplot(212)
plt.subplots_adjust(hspace = 0.5)
plt.title('Loss')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.grid(color='b', linestyle='--', linewidth=0.5)
plt.show()

# Architecture 2
# Build the ANN
classifier = Sequential()
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
#keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the ANN
history = classifier.fit(X_train_sm, y_train_sm, validation_data = (X_test, y_test), batch_size = 20, epochs = 100)


# Plot accuracy during training
plt.subplot(211)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.grid(color='b', linestyle='--', linewidth=0.5)

plt.subplot(212)
plt.subplots_adjust(hspace = 0.5)
plt.title('Loss')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.grid(color='b', linestyle='--', linewidth=0.5)
plt.show()

# Part 3 - evaluating the model

y_predRaw = classifier.predict(X_test)
y_pred1 = np.array([])
for row in y_predRaw:
    y_pred1 = np.append(y_pred1, row.argmax())


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback

cm = confusion_matrix(y_test, y_pred1)
precision_05T = precision_score(y_test, y_pred1, average = None)
recall_05T = recall_score(y_test, y_pred1, average = None)
recall_05T = np.append(recall_05T, [0])
recall_05T = np.append(recall_05T, [0])

def get_precision_1T(cm):
    precision = np.array([])
    cm = np.insert(cm,0,0,axis = 1)
    cm = np.insert(cm,7,0,axis = 1)
    for i in range(1, cm.shape[0]+1):
        column_sum = np.sum(cm.T[i])
        if column_sum != 0:
            precision_score = (cm[i-1, i] + cm[i-1, i+1] + cm[i-1, i-1])/column_sum
            if precision_score >= 1:
                precision_score = 1.0
        else:
            precision_score = 0     
        precision = np.append(precision, precision_score) 
        #print(precision)
    return precision 

precision_1T = get_precision_1T(cm)
cm = np.vstack((cm, np.transpose(precision_05T)))
cm = np.vstack((cm, precision_1T))
cm = np.vstack((cm.T, recall_05T)).T

# Part 4 - k-fold cross validation

# Evaluating the ANN using 10 fold cross validiation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train_sm, y = y_train_sm, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()



