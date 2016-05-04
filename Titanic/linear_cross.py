import numpy as np
import pandas
import sklearn

titanic = pandas.read_csv("train.csv")
titanic_test = pandas.read_csv("test.csv")

'''
==========cleaning of training data==========
'''

#cleaning Fare column by replacing it with median
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

#cleaning the age column using mean Age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())

#cleaning the embarked column
titanic["Embarked"] = titanic["Embarked"].fillna(0)

#conversion of non-numeric data to numeric data
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

''' 
    applying cross validation using linear regression model
'''
# Import the linear regression class
from sklearn.linear_model import LinearRegression

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

from sklearn import cross_validation

# Initialize our algorithm class
alg = LinearRegression()

print "Applying K-Fold cross_validation using 3 subsamples\n\n"
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print "Average score over all iterations :"
print(scores.mean())