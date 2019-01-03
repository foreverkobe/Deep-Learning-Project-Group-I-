import os

import numpy as np
from tqdm import tqdm

from data_reader import get_data

# PAPER: Specifically, we set up the maximum lag of the LSTM to include 10 successive observations
LSTM_WINDOW_SIZE = 10

# sigma, returns, comput, crcard, invest, bnkrpt are the most useful predictors.
# because we use mask, we start by sigma first then we add returns, then comput and so forth.

#Original Predictors proposed by the paper
#PREDICTORS = ['sigma',
#              'returns',
#              'Trend COMPUT',
#              'Trend CRCARD',
#              'Trend INVEST',
#              'Trend BNKRPT',
#              'Trend AIRTVL',
#              'Trend ADVERT',
#              'Trend AUTO',
#              'Trend AUTOBY',
#              'Trend AUTOFI',
#              'Trend BIZIND',
#              'Trend COMLND',
#              'Trend CONSTR',
#              'Trend DURBLE',
#              'Trend EDUCAT',
#              'Trend FINPLN',
#              'Trend FURNTR',
#              'Trend INSUR',
#              'Trend JOBS',
#              'Trend LUXURY',
#              'Trend MOBILE',
#              'Trend MTGE',
#              'Trend RENTAL',
#              'Trend RLEST',
#              'Trend SHOP',
#              'Trend SMALLBIZ',
#              'Trend TRAVEL',
#              'Trend UNEMPL']

# New x features
PREDICTORS = ['sigma',
              'returns',
              'VIX',
              'Trend COLOR',
              'Trend DEBT',
              'Trend ECONOMICS',
              'Trend HOUSING',
              'Trend INFLATION',
              'Trend MARKETS',
              'Trend REVENUE',
              'Trend STOCKS',
              'Trend UNEMPLOYMENT']

if 'DEBUG' in os.environ:
    PREDICTORS = PREDICTORS[0:5]
    print('DEBUG we truncate the predictors because we dont consider all the trends.')
    print(PREDICTORS)

INPUT_SIZE = len(PREDICTORS)

#Set array for x and y
def chunker(seq, size):
    return [(seq[pos:pos + size], seq[ pos + size - 1 :pos + size + 1]) for pos in range(0, len(seq), 1)]


def df_to_keras_format(df):
    keras_x = []
    keras_y = []

    for x, y in tqdm(chunker(df, LSTM_WINDOW_SIZE), 'df_to_keras_format'):
        # print('*' * 80)

        # filter on predictors
        x = x[PREDICTORS]
        y = y[PREDICTORS]

        # print(x)
        # print(y)

        x_new = x.values
        y_new = y['sigma'].values
        if len(x_new) == LSTM_WINDOW_SIZE and len(y_new) == 2:
            #Here we set the binary version, we only predict the direction of volatility in the next day
            #If we want to predict the value, just need to take one value from the chunker
            y_new = [float((y['sigma'].iloc[1] - y['sigma'].iloc[0]) > 0 )]
            keras_x.append(x_new)
            keras_y.append(y_new)
    
    keras_x = np.array(keras_x)
    keras_y = np.array(keras_y)
    return keras_x, keras_y

# Main function of next_batch.py
# Get x varaibles and y variable as ndarray to put in tensorflow
def get_trainable_data():
    tr, te, mean, std = get_data()
    print(tr.head())
    print(te.head())

    x_train, y_train = df_to_keras_format(tr)
    x_test, y_test = df_to_keras_format(te)

    return (x_train, y_train), (x_test, y_test), mean, std


if __name__ == '__main__':
     (x_train, y_train), (x_test, y_test), mean, std = get_trainable_data()
