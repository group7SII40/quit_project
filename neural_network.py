import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('HRDataset_v14.csv')
number = preprocessing.LabelEncoder()
data['DOB'] = number.fit_transform(data.DOB)
data['Sex'] = number.fit_transform(data.Sex)
data['DateofHire'] = number.fit_transform(data.DateofHire)
data['DateofTermination'] = number.fit_transform(data.DateofTermination)
data['TermReason'] = number.fit_transform(data.TermReason)
data['EmploymentStatus'] = number.fit_transform(data.EmploymentStatus)
data['PerformanceScore'] = number.fit_transform(data.PerformanceScore)
data['LastPerformanceReview_Date'] = number.fit_transform(data.LastPerformanceReview_Date)

# Seperating Columns into dependent and independent variables 
X=data[['Salary','TermReason','PerformanceScore', 'EmpSatisfaction']]
y=data['EmploymentStatus']
# splitting into train (70%) and test (30%) set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define and build a Sequential model, and print a summary
# The model expects rows of data with 4 variables (the input_shape=[4] argument)
# The first hidden layer has 600 nodes and uses the relu activation function.
# The second hidden layer has 200 nodes and uses the relu activation function.
# The output layer has 20 nodes and uses the softmax activation function.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[4]),
    keras.layers.Dense(600, activation="relu"),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(20, activation="softmax")
])
print(model.summary())

# compile the keras model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
# Fit the model
history = model.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_test, y_test))

#NOTES
# Keras is a high-level API that runs on top of TensorFlow
# The Sequential API is a framework for creating models based 
# on instances of the sequential() class. The model has one input 
# variable, a hidden layer with two neurons, and an output layer with
# one binary output. Additional layers can be created and added to the 
# model. The summary() function is used to generate and print the summary 
# in the Python console