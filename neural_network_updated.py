from cairo import PDF_METADATA_KEYWORDS
from sympy import O
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

#OneHotEncoding of categorical input features "TermReason" "PerfomanceScore" "EmpSatisfatcion" and output "EmploymentStatus"

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

termReason = np.array(X['TermReason']) #create numpy array
termReason = termReason.reshape(len(termReason), 1) #reshape it in order to use the fit_transform method
termReason = enc.fit_transform(termReason)

print(termReason)

performanceScore = np.array(X['PerformanceScore'])
performanceScore = performanceScore.reshape(len(performanceScore), 1)
performanceScore = enc.fit_transform(performanceScore)

print(performanceScore)

empSatisfaction = np.array(X['EmpSatisfaction'])
empSatisfaction = empSatisfaction.reshape(len(empSatisfaction), 1)
empSatisfaction = enc.fit_transform(empSatisfaction)

print(empSatisfaction)

y = np.array(y)
y = y.reshape(len(y), 1)
y = enc.fit_transform(y)

print(y)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#normalize continue features "Salary"  0 ... +infinite -> 0 ... +1

salary = np.array(X['Salary'])
salary = salary.reshape(len(salary), 1)
salary = scaler.fit_transform(salary)

print(salary)

#rebuild the X vector in order to have a flattened array

X = []
for a,b,c,d in zip(salary, termReason, performanceScore, empSatisfaction):
    x = [*a,*b,*c,*d]
    X.append(np.array(x))

X = np.array(X)


# splitting into train (70%) and test (30%) set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define and build a Sequential model, and print a summary
# The model expects rows of data with 4 variables (the input_shape=[4] argument)
# The first hidden layer has 600 nodes and uses the relu activation function.
# The second hidden layer has 200 nodes and uses the relu activation function.
# The output layer has 20 nodes and uses the softmax activation function.

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=X_train[0].shape),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])
print(model.summary())
import keras.backend as K

#define f1_metric

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

METRICS = [
      #keras.metrics.TruePositives(name='tp'),
      #keras.metrics.FalsePositives(name='fp'),
      #keras.metrics.TrueNegatives(name='tn'),
      #keras.metrics.FalseNegatives(name='fn'), 
      #keras.metrics.BinaryAccuracy(name='accuracy'),
      #keras.metrics.Precision(name='precision'),
      #keras.metrics.Recall(name='recall'),
      #keras.metrics.AUC(name='auc'),
      #keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      f1_metric
]

opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.005)

# compile the keras model
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=METRICS)
# Fit the model
history = model.fit(X_train, y_train, epochs=100, 
                    validation_data=(X_test, y_test), batch_size=len(X_train))

#NOTES
# Keras is a high-level API that runs on top of TensorFlow
# The Sequential API is a framework for creating models based 
# on instances of the sequential() class. The model has one input 
# variable, a hidden layer with two neurons, and an output layer with
# one binary output. Additional layers can be created and added to the 
# model. The summary() function is used to generate and print the summary 
# in the Python console
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, average_precision_score

def score_nn(trained_model, X_input, y_test):
    y_prob=trained_model.predict(X_input)
    y_pred=y_prob.copy()
    y_pred[y_prob<0.5]=0
    y_pred[y_prob>=0.5]=1
    acc=accuracy_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred, average="weighted")
    prec=precision_score(y_test, y_pred, average="weighted")
    auroc=roc_auc_score(y_test, y_prob,  average="weighted")
    ap=average_precision_score(y_test, y_prob, average="weighted")
    f1=f1_score(y_test, y_pred, average="weighted")
    score={'accuracy':acc, 'recall':rec, 'precision':prec, 'auroc':auroc, 'average precision':ap, 'f1':f1}
    return score

print(score_nn(model, X_test, y_test))
