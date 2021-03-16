import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from pandas import read_csv

sonar_data = read_csv('D:/CNN/sonar.all-data.csv', header=None)

sonar_data.head()
#print(sonar_data.head())
# dem so hang va cot
sonar_data.shape
#print(sonar_data.shape())

sonar_data.describe() 

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()


X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
#print(X)
#print(Y)

#Training and Test data
#validation va seed 
#stratifi 
# va random 
validation_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, stratify=Y, random_state=seed)

#.shape 
# print(X.shape, X_train.shape, X_test.shape)

# print(X_train)
# print(Y_train)

#Model Training --> Logistic Regression
model = LogisticRegression()


#fit mo hinh
model.fit(X_train, Y_train)

#Model Evaluation

#model.predict 
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 

#print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 

#print('Accuracy on test data : ', test_data_accuracy)

#Making a Predictive System

input_data = X_test.iloc[18]

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')
