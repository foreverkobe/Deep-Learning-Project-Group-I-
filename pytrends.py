# adjusted from the codes that Malik Koné posted on GitHub
# https://github.com/GeneralMills/pytrends/issues/174

# set working directory (location where code is)
import os
os.chdir("/Users/smileyue/Desktop/CU/Deep Learning/pytrends")
 
from pytrends.request import TrendReq
import pandas as pd
import datetime as dt
import math

#### connect to google
_pytrends = TrendReq(hl='en-US', tz=360)
#### build the playload
_kw_list = ['color','markets','unemployment','revenue','housing','portfolio','inflation','economics','debt','stocks']
#_kw_list = ['portfolio']
_cat = 0
_geo = ''
_gprop = '' 
# dates can be formated as  `2017-12-07 2018-01-07`, or  `today 3-m` `today 5-y`  check trends.google.com's url
_date_fmt = '%Y-%m-%d'
_start_date, _end_date = map(lambda x : dt.datetime.strptime(x, _date_fmt)
                           , ['2006-10-22', '2017-10-03'])

### Building an array of 90d periods to retrieve google trend data with a one day resolution
_90d_periods = math.ceil( (_end_date - _start_date) / dt.timedelta(days=90) ) 

# _tmp_range is a list of dates separated by 90d.  We need one more than the number of _90_periods.  if _end_date is in the future google returns the most recent data
_tmp_range = pd.date_range(start= _start_date, periods= _90d_periods + 1, freq= '90D')

# making the list of `_start_date _end_date`, strf separated by a space
_rolling_dates = [ ' '.join(map(lambda x : x.strftime(_date_fmt)
                                , [_tmp_range[i], _tmp_range[i+1] ])
                            )
                    for i in range(len(_tmp_range)-1) ]

# loop to get trend data of all the words in the list
for word in _kw_list:      
    
    # initialization of the major data frame _df_trends
    # _dates will contains our last playload argument
    _dates = _rolling_dates[0]
    _pytrends.build_payload([word], cat=_cat, timeframe=_dates, geo=_geo, gprop=_gprop)
    _df_trends= _pytrends.interest_over_time()
    
    for _dates in _rolling_dates[1:] :
        # we need to normalize data before concatenation
        _common_date = _dates.split(' ')[0]
        _pytrends.build_payload([word], cat=_cat, timeframe=_dates, geo=_geo, gprop=_gprop)
        _tmp_df =   _pytrends.interest_over_time()
        _multiplication_factor = _df_trends.loc[_common_date] / _tmp_df.loc[_common_date]
    
        _df_trends= (pd.concat([_df_trends,
                               (_tmp_df[1:]* _multiplication_factor)])
                     .drop(labels = 'isPartial', axis = 1)  # drop isPartial
                     .resample('D', closed='right').bfill()  # making sure that we have one value per day. 
                    )
    # adjust the csv file format to satisfy the needs of our model     
    _df_trends.index.names=['Date']
    _df_trends.sort_values(by='Date',ascending=False,inplace=True)    
    _df_trends.set_axis(['Close'],axis='columns',inplace=True)
    _df_trends.drop(_df_trends[_df_trends.index>'2017-10-03'].index,inplace=True)
    _df_trends.to_csv(word+'.csv', encoding='utf-8', index=True)