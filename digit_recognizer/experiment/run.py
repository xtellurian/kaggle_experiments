""" """

import logging
import time
import sys
import utils
from models import experiment_dict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict


logr = logging.getLogger(__name__)
logr.setLevel(logging.DEBUG)
sh = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
sh.setFormatter(formatter)
logr.addHandler(sh)


X_train_full, y_train_full = utils.import_training_data()



train_fraction = 0.8
X_train, X_test, y_train, y_test = train_test_split(X_train_full, 
                                                        y_train_full,
                                                        test_size=train_fraction, 
                                                        random_state=42) 


print('X_train has shape {}'.format(X_train.shape))
print('y_train has shape {}'.format(y_train.shape))

def run_experiment(num):
    start = time.time()
    logr.info('Running Experiment num={}'.format(num))
    target_model_name = 'expt_{}'.format(num)

    expt = experiment_dict[target_model_name]
    pipeline = expt['pl']
    pipeline.fit(X_train, y_train)

    cv = 3
    predictions = cross_val_predict(pipeline, X_test, y_test, cv=cv)
    logr.info('obtained accuracy = {:.2f}% with cv={}, pipeline={} '.format(
                                                    accuracy_score(y_test,predictions)*100,
                                                    cv,
                                                    pipeline)) 

    taken = time.time() - start
    logr.info('expt {} took {} seconds'.format(num, taken ))

for i in range(1, len(experiment_dict) + 1):
    run_experiment(i) 


