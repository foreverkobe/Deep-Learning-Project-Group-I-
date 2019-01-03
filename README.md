# Deep Learning Project - Group I

## Papers worked on: 

Xiong, Ruoxuan, Eric P. Nichols, and Yuan Shen. Deep Learning Stock Volatilities with Google Domestic Trends. arXiv preprint arXiv:1512.04916 (2015).https://arxiv.org/pdf/1512.04916.pdf

Preis, T., Moat, H. S., & Eugene Stanley, H. (2013) Quantifying Trading Behavior in Financial Markets Using Google Trends. Scientific Reports, 3. https://doi.org/10.1080/10888438.2015.1057824


## Original repo used:

Deep Learning Stock Volatilities with Google Domestic Trends
https://github.com/philipperemy/stock-volatility-google-trends

Google Trend Data Retrieval
https://github.com/GeneralMills/pytrends/issues/174


## Code:

run_model_tf_value.py: Runs the result of predicting VIX index value and trading VXX

run_model_tf_bin.py: Runs the result of strategy improvement part

data_reader.py: Data preprocessing

next_batch.py: Convert data into trainable format

pytrends.py: Retrieve Google Trend data.


## Jupyter Notebook:

run_model_tf_value.ipynb: same content as run_model_tf_value.py, shows snapshots of results on the report

run_model_tf_bin.ipynb: same content as run_model_tf_bin.py, shows snapshots of results on the report


## Data:

trends folder: Retrieved Google Trend data, VIX data, VXX data and S&P 500 data

data.npz: Trainable data after preprocessing, saves y variables as value of volatility

data_binary.npz: Trainable data after preprocessing, saves y variables as 0(decrease) and 1(increase)


## Report (in report folder):

PDF file: report.pdf
TEX file: report.tex


## How to run the codes:

run run_model_tf_value.py to get the result of predicting VIX index value and trading VXX using naive strategy

run run_model_tf_bin.py to get the result of trading VXX using improved strategy


