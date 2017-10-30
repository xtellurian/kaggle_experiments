"""This module contains pipeline definitions"""

import logging
import numpy as np
import sys

# set up a logger, at least for the ImportError 
model_logr = logging.getLogger(__name__)
model_logr.setLevel(logging.DEBUG)
model_sh = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
model_sh.setFormatter(formatter)
model_logr.addHandler(model_sh)

# model imports
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


experiment_dict = \
    { 
    # Note: keys are of the form expt_*, which are used to execute the 
    #   associated values of 'pl' keys     
    #
    # experiments to build pipeline ################################################
    'expt_1': { 
        'note': 'random guessing (maintains class distributions)',
        'name': 'Crash Test Dummies', 
        'pl': Pipeline([ ('dummy_clf', DummyClassifier()) ])
        },
    'expt_2': { 
        'note': 'KNeigbours',
        'name': 'K Nearest Neighbours', 
        'pl': Pipeline([ ('K_Neighbours', KNeighborsClassifier() )])
        },
    'expt_3': { 
        'note': 'Using linear kernel and C=0.025',
        'name': 'SVC Classifier', 
        'pl': Pipeline([ ('dummy_clf', SVC(kernel="linear")) ])
        }
    } # end of experiment dict

    
# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]