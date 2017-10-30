import sys
import os
import logging
import pandas as pd

# set up a logger, at least for the ImportError 
utils_logr = logging.getLogger(__name__)
utils_logr.setLevel(logging.DEBUG)
utils_sh = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
utils_sh.setFormatter(formatter)
utils_logr.addHandler(utils_sh)


def import_training_data(target_col = 'label'):
    """Loads full training data set from /data directory 
    Returns training and targets as a tuple (X, y) with target removed from X
    """
    dir = os.path.dirname(os.path.dirname(__file__)) # go up one level to get root of this experiment
    path = os.path.join(dir, 'data','train.csv')
    utils_logr.info('Loading data from {} as pandas df'.format(path))
    df = pd.read_csv(path)
    y = df[target_col]
    df = df.drop(target_col, axis=1)
    return df, y
