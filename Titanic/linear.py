import pandas
import sklearn
import numpy as np
from scipy.stats import mode

titanic = pandas.read_csv("train.csv")
titanic_test = pandas.read_csv("test.csv")

'''
==========cleaning of training data==========
'''

#cleaning Fare column by replacing it with median
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

#cleaning the age column using mean Age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

#cleaning the embarked column
titanic["Embarked"] = titanic["Embarked"].fillna(0)

#conversion of non-numeric data to numeric data
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


'''
==========cleaning of testing data==========
'''

#cleaning Fare column by replacing it with classwise mean
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
#cleaning the age column using mean Age
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

#cleaning the embarked column
titanic_test["Embarked"] = titanic_test["Embarked"].fillna(0)
    
#conversion of non-numeric data to numeric data
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

'''
==========applying Linear Regression model==========
'''

from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
# Initialize the algorithm class
alg = LinearRegression()

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    
# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
    
# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("finalSubmissionLinear.csv", index=False)


print "saved results to finalSubmissionLinear.csv"