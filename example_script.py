import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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

# print("----------COLUMN---------------", data['EmploymentStatus'])
# print(data, data['DOB'], data['Sex'],data['DateofHire'],data['DateofTermination'],data['TermReason'],data['EmploymentStatus'],data['PerformanceScore'],data['PerformanceScore'],
# data['LastPerformanceReview_Date'])

# Seperating Columns into dependent and independent variables 
X=data[['Salary','TermReason','PerformanceScore', 'EmpSatisfaction']]
y=data['EmploymentStatus']
# splitting into train (70%) and test (30%) set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Fitting Random Forest Regression to the dataset
# import the regressor
# train the model on the training set and perform predictions on the test set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)

# Train the model using the training set
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)
print("PREDICTION FOR UNKNOWN INPUT  --------->  ",classifier.predict([[500,4,3,1]]))
print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision Score of the classifier is:" ,precision_score(y_test, prediction,average='weighted'))
print("Recall Score of the classifier is:" ,recall_score(y_test, prediction,average='weighted'))
print("F1 Score of the classifier is: " , f1_score(y_test, prediction,average='weighted'))

# Make prediction for a new row