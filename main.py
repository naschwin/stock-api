############# Imports #############
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yahoo_fin.stock_info as info
import yahoo_fin.news as news
import pandas as pd
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datetime import datetime
from time import mktime

from nltk.sentiment.vader import SentimentIntensityAnalyzer


############# App Configuration #################
app = FastAPI()

origins = ['http://localhost:3000', '*']

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

############# Functions #################
# Creating an EMA indicator function
def ema(data, period=20, col='close'):
    return data[col].ewm(span=period, adjust = False).mean()

# Weighted Average
def w_avg(data):
    weights = list(range(1, len(data)+1))
    mult_wt = [i*j[0] for i, j in zip(weights, data)]
    return sum(mult_wt)/sum(weights)

# Indicators
def MACD(data, short_term=12, long_term=26, signal_term=9, col='close'):
    """
    MACD Calculation
    """
    # Short term ema
    ShortEMA = ema(data, short_term, col)
    # Long term ema
    LongEMA = ema(data, long_term, col)
    # MACD Line
    MACD = ShortEMA - LongEMA
    # Signal Line
    signal = MACD.ewm(span=signal_term, adjust=False).mean()

    return MACD, signal

def RSI(data, period=14, col='close'):
    """
    RSI Calculation
    """
    diff = data[col].diff(1)
    diff = diff.dropna()
    up = diff.copy()
    down = diff.copy()
    up[up<0] = 0
    down[down>0] = 0
    data['up'] = up
    data['down'] = down
    avg_gain = ema(data, period, col='up')
    avg_loss = abs(ema(data, period, col='down'))
    rs = avg_gain/avg_loss
    rsi = 100.0 - (100.0/(1.0+rs))

    return rsi

def stoch(data, period = 14, d_period = 3):
    """
    Stochastic Oscillator Calculation
    """
    high_period = data['high'].rolling(period).max()
    low_period = data['low'].rolling(period).min()

    k = ((data['close'] - low_period)/(high_period-low_period))*100
    d = k.rolling(d_period).mean()

    return k, d

def svr_predict(df, vals = False):
    X = df.iloc[1:-1,:]
    Y = df.iloc[1:-1, 0]

    Z = pd.DataFrame(Y)

    si=SimpleImputer(
        missing_values=np.nan,
        strategy="mean"
    )

    X=si.fit_transform(X)
    X=pd.DataFrame(X)

    Y = Y.values.reshape(len(Y), 1)

    Y = si.fit_transform(Y)

    ss=StandardScaler()

    X_s=ss.fit_transform(X)
    Y_s=ss.fit_transform(Y)

    Y = pd.DataFrame(Y)
    Y.index = Z.index

    X_train,X_test,Y_train,Y_test=train_test_split(X_s,Y_s,test_size=0.3,shuffle=False)

    svr=SVR(kernel="rbf")
    fitted_svr = svr.fit(X_s,Y_s.ravel())

    Y_pred = fitted_svr.predict(X_test)

    Y_pred=Y_pred.reshape(-1,1)

    actual_Y=ss.inverse_transform(Y_test)[:,0]
    pred_Y=ss.inverse_transform(Y_pred)[:,0]

    if pred_Y[len(actual_Y)-1] > pred_Y[len(actual_Y) - 2]:
        action = 'buy'
    else:
        action = 'sell'

    df_compare = pd.DataFrame()
    df_compare['Actual Val'] = actual_Y
    df_compare['Predicted Val'] = pred_Y
    df_compare.index = Z.index[int(len(Z)*0.7):]
    df_compare


    if not vals:
        return action
    else:
        return action, df_compare

def vader_predict(df, vals = False):
    vader = SentimentIntensityAnalyzer()
    f = lambda title: vader.polarity_scores(title)['compound']
    df['Compound'] = df['Summary'].apply(f)
    mean_df = df.groupby(['Date']).mean()

    val = mean_df.values

    mean_df = mean_df.unstack()
    mean_df = mean_df.xs('Compound').transpose()

    # mean_df.plot(kind='bar')

    if not vals:
        return val
    else:
        return val, mean_df



############ Routes ################

# Test
@app.get("/", tags=["test"])
def greet():
    return {'helo': 'nubs'}

# Stock Price
@app.get("/stock/{stock_ticker}")
def get_stock(stock_ticker, start_date = "1/1/2014"):
    data = info.get_data(stock_ticker, start_date)
    data.drop(["adjclose", "ticker"], axis=1, inplace=True)

    the_dt = { i: j for i, j in zip(data.index.astype(str), data.to_dict('records'))}
    the_dt

    return json.dumps(the_dt)


@app.get('/predict')
def predictpage():
    return {'Success': "You've reached predict page"}

# Prediction
@app.get("/predict/ml/{stock_ticker}")
def predict_ml(stock_ticker):
    data = info.get_data(stock_ticker, start_date="1/1/2014")
    data.drop(["adjclose", "ticker"], axis=1, inplace=True)

    data['Macd'], data['Signal'] = MACD(data)
    data['Rsi'] = RSI(data)
    data['%K'], data['%D'] = stoch(data)

    pcor_feature_df = data[["close", "volume", "Rsi", "Macd", "%K"]]

    action = svr_predict(pcor_feature_df)

    return {'Prediction':action}

#test
@app.get("/predict/ml")
def test():
    return {'helo': 'lol'}

# Prediction with Values
@app.get("/predict/ml/{stock_ticker}/values")
def predict_ml_val(stock_ticker):
    data = info.get_data(stock_ticker, start_date="1/1/2014")
    data.drop(["adjclose", "ticker"], axis=1, inplace=True)

    data['Macd'], data['Signal'] = MACD(data)
    data['Rsi'] = RSI(data)
    data['%K'], data['%D'] = stoch(data)

    pcor_feature_df = data[["close", "volume", "Rsi", "Macd", "%K"]]

    action, df = svr_predict(pcor_feature_df, True)

    return {'Prediction':action, 'Data': df}

# Sentiment Analysis
@app.get("/predict/senti/{stock_ticker}")
def predict_senti(stock_ticker):
    data = news.get_yf_rss(stock_ticker)
    articles = [ [i['published_parsed'], i['summary']] for i in data]

    article_df = pd.DataFrame(articles, index=range(len(articles)))
    article_df.columns = ['Date', 'Summary']
    article_df.sort_values(by=['Date'], inplace=True)

    tm = lambda dt: datetime.fromtimestamp(mktime(dt)).date()
    article_df['Date'] = article_df['Date'].apply(tm)

    vader_df = vader_predict(article_df)

    if w_avg(vader_df) <= 0:
        action = 'sell'
    else:
        action = 'buy'

    return {'Prediction':action}

# Sentiment Analysis with Values
@app.get("/predict/senti/{stock_ticker}/values")
def predict_senti_val(stock_ticker):
    data = news.get_yf_rss(stock_ticker)
    articles = [ [i['published_parsed'], i['summary']] for i in data]

    article_df = pd.DataFrame(articles, index=range(len(articles)))
    article_df.columns = ['Date', 'Summary']
    article_df.sort_values(by=['Date'], inplace=True)

    tm = lambda dt: datetime.fromtimestamp(mktime(dt)).date()
    article_df['Date'] = article_df['Date'].apply(tm)

    vader_df, values = vader_predict(article_df)

    if w_avg(vader_df) <= 0:
        action = 'sell'
    else:
        action = 'buy'

    return {'Prediction':action, 'Data': values}

# Get news
@app.get("/news/{stock_ticker}")
def get_news(stock_ticker):
    data = news.get_yf_rss(stock_ticker)
    summaries = [ i['summary'] for i in data]
    links = [ i['link'] for i in data]

    return {'Summaries': summaries, 'Links': links}





############ Start #################
if __name__ == "__main__":
    uvicorn.run("main:app", host = "localhost", port=8000, reload=True)
