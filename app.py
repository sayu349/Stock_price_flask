# Flaskライブラリ
from flask import Flask
from flask import render_template
from flask import request
from flask import send_file


import numpy as np
import pandas as pd
import yfinance as yf
#from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm

# おまじない
app = Flask(__name__,static_folder='./templates/images')

# company_nameに2つの株の銘柄名を入れて回せばOK
def calc_efficient_frontier(company_name):
    df = pd.DataFrame()

    for name in company_name:
        #adj_data = wb.DataReader(name, 'yahoo',start="2009-1-1")['Adj Close']
        adj_data = yf.download(name,start="2009-01-01")["Adj Close"]
        df[name] = adj_data

    log_returns = np.log(df / df.shift(1))
    sec_returns = log_returns.mean() * 250
    sec_std = log_returns.std() * 250
    log_returns.cov() * 250
    log_returns.corr()

    pfolio_returns = []
    pfolio_volatilities = []

    for x in range(1000): # 1000買い繰り返し
        weights = np.random.random(len(company_name)) # 2つの数値、0~1の間の乱数を発生させる
        weights /= np.sum(weights) # 全体を0~1の間でスケール調整する
        pfolio_returns.append(np.sum(weights * log_returns.mean()) * 250) # ポートフォリオの予想利益率
        pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*250, weights)))) # ポートフォリオの標準偏差

    # 行列化する
    pfolio_returns = np.array(pfolio_returns)
    pfolio_volatilities = np.array(pfolio_volatilities)

    pfolio_returns,pfolio_volatilities # 適当な割合を1000回作成して、1000回分の計算結果をそれぞれのリストに格納した。

    portfolios = pd.DataFrame({'Return' : pfolio_returns, 'Volatility' : pfolio_volatilities})

    
    portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(10,6))
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.grid()
    plt.savefig("./templates/images/img.png")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/plt_efficient_frontier',methods=["POST"])
def calc_figure():
    company_name_1 = request.form["company_name_1"] + ".T"
    company_name_2 = request.form["company_name_2"] + ".T"

    company_name_list = [company_name_1,company_name_2]
    calc_efficient_frontier(company_name_list)
    img = "img.png"
    return render_template('result.html')

@app.route('/Monte_Carlo_simulation')
def Monte_Carlo():
    return render_template("stock_price_prediction.html")

@app.route('/Monte_Carlo_simulation_results',methods=["POST"])
def calc_stock_price():
    ticker = request.form["company_name"] + ".T"
    data = pd.DataFrame()
    data[ticker] = yf.download(ticker,start='2007-1-1')['Adj Close']
    log_returns = np.log(1 + data.pct_change())
    # 現在の株価
    data.plot(figsize=(10, 6))
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    Z = norm.ppf(np.random.rand(10,2))
    t_intervals = 1000 # 1000日分の計算をする　(1000行)
    iterations = 10 # 10パターンの数値で、（10列）
    daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))
    S0 = data.iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0

    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    plt.figure(figsize=(10,6))
    plt.plot(price_list)
    plt.savefig("./templates/images/stck_price_result.png")

    return render_template("stock_price_prediction_result.html")