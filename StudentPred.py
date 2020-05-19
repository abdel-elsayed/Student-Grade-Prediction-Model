import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#reading in the data from the data-set
data = pd.read_csv("student-mat.csv", sep=";")

#getting the attributes that we want to measure
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#the final grade label that we want to predict
predict = "G3"

#return a new data set wihtout the attribute we predicting
x = np.array(data.drop([predict], 1))

#labels
y = np.array(data[predict])

#spliting 10% of the data into test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


linear = linear_model.LinearRegression()

#fitting the training dataset into a linear regression model
linear.fit(x_train, y_train)

#accuracy from the linear regression model
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)


for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


