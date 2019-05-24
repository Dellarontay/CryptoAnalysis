# Kaggle
## Keras for deep learning
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential

## Scikit learn for mapping metrics
from sklearn.metrics import mean_squared_error

#for logging
import time

##matrix math
import numpy as np
import math

##plotting
import matplotlib.pyplot as plt

##data processing
import pandas as pd

# End Kaggle


import h5py
# import scipy
# from scipy import ndimage

from PIL import Image
# from lr_utils import load_dataset
# %matplotlib inline
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

# from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def loader(infile):
    dataframe = pd.read_csv(infile, parse_dates=['Date'])
    # dataframe = dataframe.drop([0,1,2,3],axis=0)
    crypto_data = {}
    # crypto_data["0x"] = dataframe.loc[dataframe["Currency"] == "0x"]
    crypto_data["bitcoin"] = dataframe.loc[dataframe["Currency"] == "bitcoin"]
    crypto_data["ethereum"] = dataframe.loc[dataframe["Currency"] == "ethereum"]
    crypto_data["ripple"] = dataframe.loc[dataframe["Currency"] == "ripple"]
    crypto_data["litecoin"] = dataframe.loc[dataframe["Currency"] == "litecoin"]
    crypto_data["eos"] = dataframe.loc[dataframe["Currency"] == "eos"]
    crypto_data["bitcoin-cash"] = dataframe.loc[dataframe["Currency"] == "bitcoin-cash"]
    crypto_data["tron"] = dataframe.loc[dataframe["Currency"] == "tron"]
    crypto_data["stellar"] = dataframe.loc[dataframe["Currency"] == "stellar"]
    crypto_data["binance-coin"] = dataframe.loc[dataframe["Currency"] == "binance-coin"]




    # bitcoin,ethereum,ripple,litecoin,eos,bitcoin-cash,tron,stellar,binance-coin
    return crypto_data

def predict(crypto_data):

    for coin in crypto_data:
        # fit model
        df_coin = pd.DataFrame(crypto_data[coin])
        df_coin = df_coin[['Date','velocity']]
        df_coin.set_index('Date', inplace = True)


        model = ARIMA(df_coin, order=(5,1,0))
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        # plot residual errors
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        plt.show()
        residuals.plot(kind='kde')
        plt.xlabel(coin)
        plt.show()
        print(residuals.describe())
    
    # for coin in crypto_data:
    #     df = pd.DataFrame(crypto_data[coin])
    #     temp_df = pd.DataFrame()
    #     temp_df['ds'] = df['Date']
    #     temp_df['y'] = df['Close']
    #     temp_df['ds'] = temp_df['ds'].dt.to_pydatetime()
    #     model = Prophet()
    #     model.fit(temp_df)
    #     future = model.make_future_dataframe(periods = 60)
    #     forecast = model.predict(future)
    #     title_str = "predicted value of "+ coin
    #     model.plot(forecast, uncertainty=False)
    #     model.plot_components(forecast, uncertainty=False)
    # pass

def mean_squared(crypto_data):
    for coin in crypto_data:

        df_coin = pd.DataFrame(crypto_data[coin])
        df_coin = df_coin[['Date','velocity']]
        df_coin.set_index('Date', inplace = True)
        X = df_coin.values
        # print(X)
        # input('here')
        size = int(len(X) * 0.80)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()

        for t in range(len(test)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            #print('predicted=%f, expected=%f' % (yhat, obs))
        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)
        # plot
        plt.plot(test)
        plt.plot(predictions, color='red')
        plt.xlabel(coin)
        plt.show()


def main():
    data = loader('clean_crypto_data.csv')

    # parameters =  initialize_parameters(200*4*365,5,10)

    # predict(data)
    mean_squared(data)
    # print(data)




    # print(data.head())

if __name__ == '__main__':
    main()
# functions: forward_propagation, compute_cost, backward_propagation, update_parameters, nn_model, predict

