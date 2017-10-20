import tools
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor, Lars, LarsCV, Perceptron, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from sklearn import linear_model

training_data = pd.read_csv('data/train.csv')

year_columns = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"]

columnsToEncode = ["MSZoning", "Street", "Alley","LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st","Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

# print("Data Description: {0} \n".format(training_data.describe()))

def preprocess(df, targetName):
    le = LabelEncoder()
    for name in columnsToEncode:
        # print("found {0} null values in {1}".format(df[name].isnull().sum(), name))
        df[name] = df[name].fillna("xxxx")
        df[name] = le.fit_transform(df[name])
    # move years to since 1900
    for name in year_columns:
        df[name] = df[name].apply(tools.years_since_1900)
    for name in df.columns:
        s = df[name]
        nulls = s.isnull().sum()
        if(nulls > 0):
            mean = s.mean()
            print("{0} has {1} null value(s). Replacing with mean {2}".format(name, nulls, mean))
            df[name] = s.fillna(mean)
    newdf = df.dropna()
    print("Dropping {0} rows due to NA".format(len(df)- len(newdf)))
    df = newdf
    # separate target
    y = df[targetName]
    x = df.drop(targetName, axis=1)
    # normalize data
    scaler = MinMaxScaler() 
    scaled_values = scaler.fit_transform(x) 
    x.loc[:,:] = scaled_values
    return x, y


X, y = preprocess(training_data, "SalePrice")

print(X.describe())

# feature selection
# remove invariate
sel = VarianceThreshold()
sel = sel.fit(X,y)
X = sel.transform(X)
# select k best
sel = SelectKBest(f_regression, k = 20)
sel = sel.fit(X, y);
X = sel.transform(X)
# print(X)
#tools.summarise_data(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)


print("Data Shape: {0}, Training size: {1} , Test size: {2}".format(X.shape,len(y_train), len(y_test)))


names = ["Linear Regression", "ARD Regression","Bayesian Ridge","ElasticNet", "Hubor Regressor",
            "Lars", "LarsCV", "Linear Regression", "Logistic Regression", "Perceptron", "Ridge",
            "SGD Regressor", "Decision Tree Regressor"]



estimators = [

    LinearRegression(),
    ARDRegression(), 
    BayesianRidge(), 
    ElasticNet(max_iter=10000),
    HuberRegressor(), 
    Lars(), 
    LarsCV(),
    LinearRegression(),
    LogisticRegression(),
    Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle=False),
    Ridge(),
    SGDRegressor(),
    DecisionTreeRegressor(max_depth=7)]


results = {}
for name, clf in zip(names, estimators):
    print("Running {0}...".format(name))
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    results[name] = score
    # print("{0} : {1}".format(name, score))

best = max(results, key=results.get)
print("{0} has R2 score of {1:.2f}".format(best, results[best]))