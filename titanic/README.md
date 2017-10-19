## Kaggle Titanic

[View Experiment](https://www.kaggle.com/c/titanic)

Data is contained in the /data directory

Classifier scripts (compares various classifiers) is contained in experiment/classify_survivors.py

## Dependencies

* [Anaconda](https://conda.io/docs/) 
* sci-kit learn `conda install scikit-learn`
* numpy `conda install -c anaconda numpy`
* matplotlib `conda install -c conda-forge matplotlib`
* pandas `conda install pandas`


## Getting Started

1) Create a new conda env `conda create --name myenv`
2) Activate that environment `activate myenv`
3) Install above dependencies
4) From the `titanic` directory, run the script from terminal:  `python /experiment/classify_survivors.py`


# Goal

It is your job to predict if a passenger survived the sinking of the Titanic or not. 
For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.

## Metric

Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.

## Submission File Format

You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.

The file should have exactly 2 columns:
```
PassengerId (sorted in any order)
Survived (contains your binary predictions: 1 for survived, 0 for deceased)
PassengerId,Survived
 892,0
 893,1
 894,0
 Etc.
 ```

 # Output
```
C:\...\kaggle\titanic (master)
> python experiment\classify_survivors.py

Survived has 2 unique values
Pclass has 3 unique values
Sex has 2 unique values
Age has 89 unique values
SibSp has 7 unique values
Parch has 7 unique values
Fare has 248 unique values
Cabin has 5 unique values
Embarked has 4 unique values
CabinClassification has 9 unique values
Dropping 0 rows due to missing values
Using 9 Features

Random Forest has R2 score of 0.85
```