from datetime import datetime


# Kaggle
## Keras for deep learning
import keras
from keras.layers import Dense, Activation, Dropout
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

def load_data(filename, sequence_length):
     #Read the data file
    raw_data = pd.read_csv(filename)
    
    train_x = raw_data.drop(['Symbol','High','Low','Volume USD','Open','Close'],axis=1).values
    # print(raw_data)
    # input("wait")
    train_y =  raw_data.drop(['Symbol','Low','Volume BTC','Volume USD','Open','Close'],axis=1).values

    print(train_x.shape)
    print(train_y.shape)
    # input("wait")
    # for data in raw_data:
    #     formed_data = str(data).strip("[]").split()
        # datetime_object = datetime.strptime(formed_data[0]., "%Y-%m-%d")

        # print(datetime_object)

        # input("wait")

    #Change all zeros to the number before the zero occurs
    # for x in range(0, raw_data.shape[0]):
    #     for y in range(0, raw_data.shape[1]):
    #         if(raw_data[x][y] == 0):
    #             raw_data[x][y] = raw_data[x-1][y]
    
    #Convert the file to a list
    data = train_x.tolist()
    
    #Convert the data to a 3D array (a x b x c) 
    #Where a is the number of days, b is the window size, and c is the number of features in the data file
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
  
    
    #Normalizing data by going through each window
    #Every value in the window is divided by the first value in the window, and then 1 is subtracted
    d0 = np.array(result)
    dr = np.zeros_like(d0)
    print(d0)
    print(dr.shape)
    # input("wait")
    # dr[:,1:] = d0[:,1:] / d0[:,0:1] - 1
    
    #Keeping the unnormalized prices for Y_test
    #Useful when graphing bitcoin price over time later
    start = 2400
    end = int(dr.shape[0] + 1)
    unnormalized_bases = d0[start:end,0:1,:]
    
    #Splitting data set into training (First 90% of data points) and testing data (last 10% of data points)
    split_line = round(0.9 * dr.shape[0])
    training_data = train_x[:int(split_line), :]
    training_data_y = train_y[:int(split_line),:]
    
    #Shuffle the data
    np.random.shuffle(training_data)
    
    #Training Data
    X_train = training_data[:, :-1]
    Y_train = y_train
    # Y_train = Y_train[:, 7]
    
    #Testing data
    X_test = dr[int(split_line):, :-1]
    Y_test = dr[int(split_line):, 49, :]
    Y_test = Y_test[:, 7]

    #Get the day before Y_test's price
    Y_daybefore = dr[int(split_line):, 48, :]
    Y_daybefore = Y_daybefore[:, 7]
    
    #Get window size and sequence length
    sequence_length = sequence_length
    window_size = sequence_length - 1 #because the last value is reserved as the y value
    
    return X_train, Y_train, X_test, Y_test, Y_daybefore, unnormalized_bases, window_size

def init_model(window_size,dropout_value,activation_function,loss_function,optimizer,X_train):
     #Create a Sequential model using Keras
    model = Sequential()

    #First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, X_train.shape[-1]),))
    model.add(Dropout(dropout_value))

    #Second recurrent layer with dropout
    model.add(Bidirectional(LSTM((window_size*2), return_sequences=True)))
    model.add(Dropout(dropout_value))

    #Third recurrent layer
    model.add(Bidirectional(LSTM(window_size, return_sequences=False)))

    #Output layer (returns the predicted value)
    model.add(Dense(units=1))
    
    #Set activation function
    model.add(Activation(activation_function))

    #Set loss function and optimizer
    model.compile(loss=loss_function, optimizer=optimizer)
    
    return model

def fit_model(model,X_train,Y_train,batch_num,num_epoch,val_split):
     #Record the time the model starts training
    start = time.time()

    #Train the model on X_train and Y_train
    model.fit(X_train, Y_train, batch_size= batch_num, nb_epoch=num_epoch, validation_split= val_split)

    #Get the time it took to train the model (in seconds)
    training_time = int(math.floor(time.time() - start))
    return model, training_time

def test_model(model, X_test,Y_test,unnormalized_bases):
    #Test the model on X_Test
    y_predict = model.predict(X_test)

    #Create empty 2D arrays to store unnormalized values
    real_y_test = np.zeros_like(Y_test)
    real_y_predict = np.zeros_like(y_predict)

    #Fill the 2D arrays with the real value and the predicted value by reversing the normalization process
    for i in range(Y_test.shape[0]):
        y = Y_test[i]
        predict = y_predict[i]
        real_y_test[i] = (y+1)*unnormalized_bases[i]
        real_y_predict[i] = (predict+1)*unnormalized_bases[i]

    #Plot of the predicted prices versus the real prices
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_title("Bitcoin Price Over Time")
    plt.plot(real_y_predict, color = 'green', label = 'Predicted Price')
    plt.plot(real_y_test, color = 'red', label = 'Real Price')
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time (Days)")
    ax.legend()
    
    return y_predict, real_y_test, real_y_predict, fig

def price_change(Y_daybefore,Y_test,y_predict):
    #Reshaping Y_daybefore and Y_test
    Y_daybefore = np.reshape(Y_daybefore, (-1, 1))
    Y_test = np.reshape(Y_test, (-1, 1))

    #The difference between each predicted value and the value from the day before
    delta_predict = (y_predict - Y_daybefore) / (1+Y_daybefore)

    #The difference between each true value and the value from the day before
    delta_real = (Y_test - Y_daybefore) / (1+Y_daybefore)

    #Plotting the predicted percent change versus the real percent change
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Percent Change in Bitcoin Price Per Day")
    plt.plot(delta_predict, color='green', label = 'Predicted Percent Change')
    plt.plot(delta_real, color='red', label = 'Real Percent Change')
    plt.ylabel("Percent Change")
    plt.xlabel("Time (Days)")
    ax.legend()
    plt.show()
    
    return Y_daybefore, Y_test, delta_predict, delta_real, fig

def binary_price(delta_predict,delta_real):
    #Empty arrays where a 1 represents an increase in price and a 0 represents a decrease in price
    delta_predict_1_0 = np.empty(delta_predict.shape)
    delta_real_1_0 = np.empty(delta_real.shape)

    #If the change in price is greater than zero, store it as a 1
    #If the change in price is less than zero, store it as a 0
    for i in range(delta_predict.shape[0]):
        if delta_predict[i][0] > 0:
            delta_predict_1_0[i][0] = 1
        else:
            delta_predict_1_0[i][0] = 0
    for i in range(delta_real.shape[0]):
        if delta_real[i][0] > 0:
            delta_real_1_0[i][0] = 1
        else:
            delta_real_1_0[i][0] = 0    

    return delta_predict_1_0, delta_real_1_0

def find_false_positives(delta_predict_1_0,delta_real_1_0):
     #Finding the number of false positive/negatives and true positives/negatives
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i in range(delta_real_1_0.shape[0]):
        real = delta_real_1_0[i][0]
        predicted = delta_predict_1_0[i][0]
        if real == 1:
            if predicted == 1:
                true_pos += 1
            else:
                false_neg += 1
        elif real == 0:
            if predicted == 0:
                true_neg += 1
            else:
                false_pos += 1
    return true_pos, false_pos, true_neg, false_neg

def calculate_statistics(ture_pos,false_pos,true_neg,false_neg,y_predict,Y_test):
    precision = float(true_pos) / (true_pos + false_pos)
    recall = float(true_pos) / (true_pos + false_neg)
    F1 = float(2 * precision * recall) / (precision + recall)
    #Get Mean Squared Error
    MSE = mean_squared_error(y_predict.flatten(), Y_test.flatten())

    return precision, recall, F1, MSE




# End Kaggle

def loader(infile):
    dataframe = pd.read_csv(infile, parse_dates=['Date'])
    # dataframe = dataframe.drop([0,1,2,3],axis=0)
    crypto_data = {}
    # crypto_data["0x"] = dataframe.loc[dataframe["Currency"] == "0x"]
    crypto_data["bitcoin"] = dataframe.loc[dataframe["Currency"] == "bitcoin"]
    # crypto_data["ethereum"] = dataframe.loc[dataframe["Currency"] == "ethereum"]
    # crypto_data["ripple"] = dataframe.loc[dataframe["Currency"] == "ripple"]
    # crypto_data["litecoin"] = dataframe.loc[dataframe["Currency"] == "litecoin"]
    # crypto_data["eos"] = dataframe.loc[dataframe["Currency"] == "eos"]
    # crypto_data["bitcoin-cash"] = dataframe.loc[dataframe["Currency"] == "bitcoin-cash"]
    # crypto_data["tron"] = dataframe.loc[dataframe["Currency"] == "tron"]
    # crypto_data["stellar"] = dataframe.loc[dataframe["Currency"] == "stellar"]
    # crypto_data["binance-coin"] = dataframe.loc[dataframe["Currency"] == "binance-coin"]




    # bitcoin,ethereum,ripple,litecoin,eos,bitcoin-cash,tron,stellar,binance-coin
    return crypto_data

# def predict(crypto_data):

#     for coin in crypto_data:
#         # fit model
#         df_coin = pd.DataFrame(crypto_data[coin])
#         df_coin = df_coin[['Date','velocity']]
#         df_coin.set_index('Date', inplace = True)


#         model = ARIMA(df_coin, order=(5,1,0))
#         model_fit = model.fit(disp=0)
#         print(model_fit.summary())
#         # plot residual errors
#         residuals = pd.DataFrame(model_fit.resid)
#         residuals.plot()
#         plt.show()
#         residuals.plot(kind='kde')
#         plt.xlabel(coin)
#         plt.show()
#         print(residuals.describe())
    
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
    # data = loader('clean_crypto_data.csv')
    # predict(data)
    # mean_squared(data)
    X_train, Y_train, X_test, Y_test, Y_daybefore, unnormalized_bases, window_size = load_data("Coinbase_btc.csv", 50)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    print(Y_daybefore.shape)
    print(unnormalized_bases.shape)
    print(window_size)

    model = init_model(window_size, 0.2, 'linear', 'mse', 'adam',X_train)
    print(model.summary())

    model, training_time = fit_model(model, X_train, Y_train, 1024, 100, .05)
    #Print the training time
    print("Training time", training_time, "seconds")


    y_predict, real_y_test, real_y_predict, fig1 = test_model(model, X_test, Y_test, unnormalized_bases)
    #Show the plot
    plt.show(fig1)

    Y_daybefore, Y_test, delta_predict, delta_real, fig2 = price_change(Y_daybefore, Y_test, y_predict)
    #Show the plot
    plt.show(fig2)

    delta_predict_1_0, delta_real_1_0 = binary_price(delta_predict, delta_real)
    print(delta_predict_1_0.shape)
    print(delta_real_1_0.shape)

    true_pos, false_pos, true_neg, false_neg = find_positives_negatives(delta_predict_1_0, delta_real_1_0)
    print("True positives:", true_pos)
    print("False positives:", false_pos)
    print("True negatives:", true_neg)
    print("False negatives:", false_neg)

    precision, recall, F1, MSE = calculate_statistics(true_pos, false_pos, true_neg, false_neg, y_predict, Y_test)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", F1)
    print("Mean Squared Error:", MSE)

if __name__ == '__main__':
    main()
# functions: forward_propagation, compute_cost, backward_propagation, update_parameters, nn_model, predict

