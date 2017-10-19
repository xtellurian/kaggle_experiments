import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import math
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# load training data

training_data = pd.read_csv('data/train.csv')

test_data = pd.read_csv('data/test.csv')

target = "Survived"
# select the columns we want to use as features (remove those below, encode sex)
columns = training_data.columns.tolist()
# drop the columns with names below
columns = [c for c in columns if c not in ["Name", "Ticket", "PassengerId"]]

# simple function for classifying cabins - encode the highest letter
def classifyCabin(x):
    if not isinstance(x, str):
        return 0
    if "A" in x:
        return 1
    if "B" in x:
        return 2
    if "C" in x:
        return 3
    if "D" in x:
        return 4
    if "E" in x:
        return 5
    if "F" in x:
        return 6
    if "G" in x:
        return 7
    return 8

def numCabins(x):
    if isinstance(x, str):
         return len(x.split())
    else:
        return 0

def preprocess(df, columns, target):
    le = LabelEncoder()
    # encode data for sex,
    df["Sex"] = le.fit_transform(df["Sex"])
    # replace unknown embarked with 'x'
    df["Embarked"] = df["Embarked"].fillna("x")
    # numerically encode embarked 
    df["Embarked"] = le.fit_transform(df["Embarked"])
    #classify cabins
    columns.append("CabinClassification")
    df["CabinClassification"] = df.Cabin.apply(classifyCabin)
    # count cabins
    df["Cabin"] = df.Cabin.apply(lambda x: numCabins(x))
    df = df[columns]
    # replace missing ages with mean
    averageAge = df["Age"].mean()
    df["Age"] = df["Age"].fillna(averageAge)
    # print range of vals in every col
    for col in df:
        print("{0} has {1} unique values".format(col ,len(df[col].unique())))
    dfnew = df.dropna() # lose all rows with missing values
    print("Dropping {0} rows due to missing values".format(len(df) - len(dfnew)))
    df = dfnew
    y = df[target] # target
    return df.drop(target, axis= 1), y


X, y = preprocess(training_data, columns, target)
print("Using {0} Features".format(len(X.columns)))

X_train, X_test, y_train, y_test = train_test_split(X, y)

names = ["Perceptron", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    linear_model.Perceptron(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=8),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


results = {}
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    results[name] = score
    # print("{0} : {1}".format(name, score))


best = max(results, key=results.get)
print("{0} has R2 score of {1:.2f}".format(best, results[best]))