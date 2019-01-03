import os
from glob import glob

import numpy as np
import pandas as pd

#import quandl

'''
np.set_printoptions(threshold=np.nan)
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
'''

# Set the path to the directory saving files
path = os.getcwd()+'/'
    # 06-Oct-2004 to 9-Apr-2015 while the test set ranges from 12-Apr-2015 to 06-Oct-2017

# Read Vix Index value from CSV file
def read_vix():
    if 'VIX.csv' in os.listdir(path+'trends/'):
        vix = pd.read_csv( path + 'trends/VIX.csv', parse_dates=True, index_col=0)
    else:
        print("No file for VIX")
#        quandl.ApiConfig.api_key = "5qhvApLaQjf-aN56focv"
#        vix = quandl.get("CBOE/VIX", start_date="2004-10-06", end_date="2017-10-05")
#        vix.to_csv(path + 'trends/VIX.csv')
        
    return vix['Close']

# Read VXX Index value from CSV file
def read_vxx():
    vxx = pd.read_csv( path + 'trends/VXX.csv', parse_dates=True, index_col=0)
        
    return vxx['Close']

# Read S&P500 Index value from CSV file
# And record S&P500 return, VXX(treat it as Volatility) and VIX index(treat it as a feature)
def read_sp_500():
    spy = pd.read_csv( path + 'trends/SP500.csv', parse_dates=True, index_col=0)
    spy['Adj Close Log'] = spy['Adj Close'].apply(np.log)
    spy['Adj Close Log Shift 1'] = spy['Adj Close Log'].shift(1)
    spy['returns'] = spy['Adj Close Log'] - spy['Adj Close Log Shift 1']

#    print(spy.head())
    
    spy.dropna(inplace=True)

    spy.drop('Adj Close Log', 1, inplace=True)
    spy.drop('Adj Close Log Shift 1', 1, inplace=True)

#    print('')
#    print(spy.head())

#    spy['u'] = (spy['High'] / spy['Open']).apply(np.log)
#    spy['d'] = (spy['Low'] / spy['Open']).apply(np.log)
#    spy['c'] = (spy['Close'] / spy['Open']).apply(np.log)
#
#    print('')
#    print(spy.head())
#    # correct.
#    # 0.511*(0.006651--0.001358)^2-0.019*(0.006651*(0.006651+-0.001358)-2*0.006651*(-0.001358))-0.383*0.006651^2
#    # ~ 0.00001482322429 [first value]
#    spy['sigma'] = 0.511 * (spy['u'] - spy['d']) ** 2 - 0.019 * (
#            spy['c'] * (spy['u'] + spy['d']) - 2 * spy['u'] * spy['d']) - 0.383 * spy['c'] ** 2
    
    spy['sigma'] = read_vxx()
    spy['VIX'] = read_vix()
#    print('')
#    print(spy.head())

#    spy.drop('u', 1, inplace=True)
#    spy.drop('d', 1, inplace=True)
#    spy.drop('c', 1, inplace=True)

    spy = spy[['returns', 'sigma', 'VIX']]  # only keep returns and volatility

#    print('')
#    print(spy.head())

    # import matplotlib.pyplot as plt
    # spy['sigma'].plot()
    # plt.show()
    return spy

#Read Google Trend Data
def process_trend(trend):
    trend_name = trend.split('/')[1].split('.')[0]
    t = pd.read_csv(trend, parse_dates=True, index_col=0)
    t = t[['Close']]
    t.columns = ['Trend {}'.format(trend_name.upper())]
#    print('Trend [{0}] processed.'.format(trend_name))
    # print(t.head())
    return t


def read_trends():
    trends = glob('trends/*.csv')
    trends = filter(lambda x: 'SP500' not in x and 'spy' not in x and 'VXX' not in x and 'VIX' not in x, trends)
    trends = sorted(trends)  # reproducibility.
    #assert len(trends) == 27, 'You should have 27 trends. Check here https://finance.google.com/finance/domestic_trends'
    trends_df_list = []
    for i, trend in enumerate(trends):
        if 'DEBUG' in os.environ and i > 2:
            print('DEBUG! WE TRUNCATE TO ONLY THREE TRENDS FOR SPEED')
            break
        trends_df_list.append(process_trend(trend))
    full_trend_df = pd.DataFrame(pd.concat(trends_df_list, axis=1))
    if full_trend_df.isnull().values.any():
#        print(full_trend_df[full_trend_df.isnull().any(axis=1)])
        full_trend_df.fillna(method='ffill', inplace=True)

        if full_trend_df.isnull().values.any():
            full_trend_df.fillna(method='bfill', inplace=True)  # we cheat a very bit at the beginning.
            # we fill gap values by the last values. If we remove lines, we will have gaps in our data
            # and it's not going to be cool.

    assert not full_trend_df.isnull().values.any()
    return full_trend_df

#Inner Join S&P500 value and Google Trend Data
def read_all():
    trends = read_trends()
    sp500 = read_sp_500()
    full_data = sp500.join(trends, how='outer')  # correct
    print('full_data',len(full_data))

#    print('-' * 80)
#    print(full_data.tail())
#    print('-' * 80)

    full_data.dropna(inplace=True)  # correct
    print('full_data',len(full_data))


#    print('-' * 80)
#    print(full_data.tail())
#    print('-' * 80)
    # print(read_sp_500().join(process_trend(trend), how='outer'))
    return full_data


def split_training_test(df):
    # 06-Oct-2004 to 9-Apr-2015 while the test set ranges from 12-Apr-2015 to 06-Oct-2017
    training_df = df[:'9-Apr-2015']
    testing_df = df['12-Apr-2015':]    
    
    ##Below is code for implmenting rolling window
    ##We roll for every 100 days
    
    #Now change to last 700 days for test
    #Length of Every rolling window is 100 right now
    #i: from 0 to 6
#    print(len(df),i*100,2058+i*100,2058+(i+1)*100)
#    training_df = df.iloc[i*100:2059+i*100,:]
#    testing_df = df.iloc[2059+i*100-10:2059+(i+1)*100,:] #10 allows continuity for window size
    return training_df, testing_df

#Standardize feature
def z_score(x, mean, std):
    return (x - mean.values) / std.values  # testing set.

#Revert back standardized data
def z_score_inv(x, mean, std):
    return x * std + mean


def apply_z_score_to_data_frame(df, mean, std):
    df = df.apply(lambda x: z_score(x, mean, std), axis=1)

    # Tips: Use this to debug.
    # print(df.head())
    # print(df.apply(lambda x: z_score(x, mean, std), axis=1).head())
    # print(mean)
    # print(std)
    return df

#Smooth data using a 3-day framework, here we set day equal to 1 to avoid smoothing
#To match trade data in the test period(otherwise we cannot revert to orignal data)
def apply_delta_t_to_data_frame(df):
    # we only support those values. They are given by the paper.
    delta_t = 1

    # for trends (MEAN)
    out = df.rolling(window=delta_t, center=False).mean()  # correct.

    # for returns (SUM)
    if 'returns' in df.columns.values:
        out['returns'] = df['returns'].rolling(delta_t, center=False).apply(np.sum)  # correct

    # for volatility (SQUARE - SUM - SQRT)
    out['sigma'] = df['sigma'].apply(np.square).rolling(delta_t, center=False).apply(np.sum).apply(np.sqrt)  # correct

    out.dropna(inplace=True)
    return out

#Main function of data_reader.py
#Here prepare original dataframe for train data and test data
def get_data():
    df = read_all()
    df = apply_delta_t_to_data_frame(df)  # try to apply z-score before and after.
    mean = np.mean(df)
    std = np.std(df)
    df = apply_z_score_to_data_frame(df, mean, std)
    tr, te = split_training_test(df)  # we cheat a bit but very little, no problem.
    return tr, te, mean['sigma'], std['sigma']


if __name__ == '__main__':
    tr,te,_,_ = get_data()
