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

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keras.utils import plot_model




def load_data(filename, sequence_length):
     #Read the data file
    raw_data = pd.read_csv(filename)
    
    training_data = raw_data.drop(['Symbol','Low','Volume USD','Open','Close'],axis=1)
    # train_x = raw_data.drop(['Symbol','High','Low','Volume USD','Open','Close'],axis=1).values
    # print(raw_data)
    # input("wait")
    # train_y =  raw_data.drop(['Symbol','Low','Volume BTC','Volume USD','Open','Close'],axis=1).values

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
    # data = training_data.tolist()
    
    #Convert the data to a 3D array (a x b x c) 
    #Where a is the number of days, b is the window size, and c is the number of features in the data file
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
  
    
    #Normalizing data by going through each window
    #Every value in the window is divided by the first value in the window, and then 1 is subtracted
    d0 = np.array(result)
    dr = np.zeros_like(d0)
    # print(d0)
    # print(dr.shape)
    # input("wait")
    dr[:,1:] = d0[:,1:] / d0[:,0:1] - 1
    
    #Keeping the unnormalized prices for Y_test
    #Useful when graphing bitcoin price over time later
    start = 2400
    end = int(dr.shape[0] + 1)
    unnormalized_bases = d0[start:end,0:1,:]
    
    #Splitting data set into training (First 90% of data points) and testing data (last 10% of data points)
    split_line = round(0.9 * dr.shape[0])
    training_data = dr[:int(split_line), :]
    training_data_y = dr[:int(split_line),:]
    
    #Shuffle the data
    np.random.shuffle(training_data)
    
    #Training Data
    X_train = training_data[:, :-1]
    Y_train = Y_train
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
       # Initialising the RNN
    model = Sequential()

    # Adding the output layer
    # model.add(Dense(1,activation='tanh'))
    #First recurrent layer with dropout
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
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
    model.compile(loss=loss_function, optimizer=optimizer,metrics=["mse"])
    
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

def calculate_statistics(true_pos,false_pos,true_neg,false_neg,y_predict,Y_test):
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
    # bitcoin,ethereum,ripple,litecoin,eos,bitcoin-cash,tron,stellar,binance-coin
    return crypto_data


def plotTrainingInfo(history):
    # Plot training Details
    # Plot MSE values
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])

    plt.title('MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['MSE',"Val MSE"], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plotFeatures(features,targets,featureList,targetList):
    plt.plot(features)
    plt.title('Features')
    plt.xlabel("Feature Value")
    plt.ylabel('Days')
    plt.legend(featureList,loc='upper left')
    plt.show()
    
    plt.plot(targets)
    plt.title('Target Values')
    plt.xlabel("Days")
    plt.ylabel('Target Value')
    plt.legend(targetList,loc='upper left')
    plt.show()


def loadData2(filename,window_length):
    raw_data = pd.read_csv(filename,header=0)
    # print(filename)
    
    # training_data = raw_data[['High','Volume BTC']]
    # training_dataY = raw_data['High']
    training_data = raw_data[['average price','velocity']]
    training_dataY = raw_data[['average price']]
    # plotFeatures(training_data[:],training_dataY[:],['average price','velocity'],['average price'])

    
    training = training_data.values
    split = int(len(training)*.9)
    # Data preprocess
    X_train = training[:split]
    X_test = training[split:]
    Y_train = training_dataY[:split]
    Y_test = training_dataY[split:]
    origY_Test= Y_test

    # unnormalized_bases = newTest[:newTest.shape[0],0:1]

    sc = MinMaxScaler() #Normalize to 1
    # sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    Y_train = sc.fit_transform(Y_train)
    X_test = sc.fit_transform(X_test)
    Y_test = sc.fit_transform(Y_test)

    # plotFeatures(X_train[:],Y_train[:],['average price','velocity'],['average price'])

    X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
    Y_train = np.reshape(Y_train,(Y_train.shape[0],Y_train.shape[1]))
    X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
    Y_test = np.reshape(Y_test,(Y_test.shape[0],Y_test.shape[1]))

    regressor = init_model(window_length, 0.2, 'tanh', 'mse', 'adam',X_train)
    regressor.summary()

    print('Training')
    BATCH_SIZE=64
    EPOCHS = 75
    history = regressor.fit([X_train], Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.05)

    plotTrainingInfo(history)

    print('Evaluation')
    loss, mse = regressor.evaluate([X_test], Y_test,
                            batch_size=BATCH_SIZE)
    print('Test loss / test mse = {:.4f} / {:.4f}'.format(loss, mse))

    # Fitting the RNN to the Training set
    # regressor.fit(X_train, Y_train, batch_size = 5, epochs = 100)
    # def fit_model(model,X_train,Y_train,batch_num,num_epoch,val_split):
    # make predictions
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(training_data)
    # trainPredict = regressor.predict(X_train)
    # standard_scalerX.fit(X_test)
    # X_test_std = standard_scalerX.transform(X_test)
    # X_test_std = X_test_std.astype('float32')
    testPredict = regressor.predict(X_test)
    testPredict = sc.inverse_transform(testPredict)
    Y_test = origY_Test
    # invert predictions
    # trainPredict2 = scaler.inverse_transform(trainPredict)
    # trainY = scaler.inverse_transform([Y_train])
    # testPredict = scaler.inverse_transform(testPredict)
    # testY = scaler.inverse_transform([Y_test])
    # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(Y_train[0], trainPredict2[:,0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(Y_test[:], testPredict[:]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    return regressor,testScore


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def main():
    # data = loader('clean_crypto_data.csv')
    # model = init_model(window_size, 0.2, 'linear', 'mse', 'adam',X_train)
    # print(model.summary())

    # model, training_time = fit_model(model, X_train, Y_train, 1024, 100, .05)
    # #Print the training time
    # print("Training time", training_time, "seconds")


    # y_predict, real_y_test, real_y_predict, fig1 = test_model(model, X_test, Y_test, unnormalized_bases)
    # #Show the plot
    # plt.show(fig1)

    # Y_daybefore, Y_test, delta_predict, delta_real, fig2 = price_change(Y_daybefore, Y_test, y_predict)
    # #Show the plot
    # plt.show(fig2)

    # delta_predict_1_0, delta_real_1_0 = binary_price(delta_predict, delta_real)
    # print(delta_predict_1_0.shape)
    # print(delta_real_1_0.shape)

    # true_pos, false_pos, true_neg, false_neg = find_positives_negatives(delta_predict_1_0, delta_real_1_0)
    # print("True positives:", true_pos)
    # print("False positives:", false_pos)
    # print("True negatives:", true_neg)
    # print("False negatives:", false_neg)

    # precision, recall, F1, MSE = calculate_statistics(true_pos, false_pos, true_neg, false_neg, y_predict, Y_test)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 score:", F1)
    # print("Mean Squared Error:", MSE)
    model, training_time = loadData2("btc_spec.csv",50)
    # plot_model(model, to_file='model.png')

    



if __name__ == '__main__':
    main()
# functions: forward_propagation, compute_cost, backward_propagation, update_parameters, nn_model, predict

