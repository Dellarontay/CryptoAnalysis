import numpy as np
import matplotlib.pyplot as plt
import h5py

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import pandas as pd

# from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean

def loader(infile):
    dataframe = pd.read_csv(infile, parse_dates=['Date'])
    # dataframe = dataframe.drop([0,1,2,3],axis=0)
    crypto_data = {}
    # crypto_data["0x"] = dataframe.loc[dataframe["Currency"] == "0x"]
    crypto_data["bitcoin"] = dataframe
    return crypto_data

def predict(crypto_data):

    for coin in crypto_data:
        # fit model
        df_coin = pd.DataFrame(crypto_data[coin])
        df_coin = df_coin[['Date','average price']]
        df_coin.set_index('Date', inplace = True)


        model = ARIMA(df_coin, order=(5,1,0))
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        print(coin)
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
        df_coin = df_coin[['Date','average price']]
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
        # newError = mean_squared_error(test, predictions)
        error = rmse(predictions,test)
        print('Test RMSE: %.3f with predictions in red' % error)
        # plot
        plt.plot(test)
        plt.plot(predictions, color='red')
        plt.xlabel(coin)
        plt.show()


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def main():
    data = loader('btc_spec.csv')
    # parameters =  initialize_parameters(200*4*365,5,10)
    # predict(data)
    mean_squared(data)
    # print(data)




    # print(data.head())

if __name__ == '__main__':
    main()
# functions: forward_propagation, compute_cost, backward_propagation, update_parameters, nn_model, predict

