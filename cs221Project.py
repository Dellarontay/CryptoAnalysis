from datetime import datetime


# Kaggle
## Keras for deep learning
import keras
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM as cudaLSTM
# from keras.layers import CuDNNLSTM as cudaLSTM
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
import keras.backend as K




def init_model(window_size,dropout_value,activation_function,loss_function,optimizer,X_train):
     #Create a Sequential model using Keras
    #    keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)

    model = Sequential()

    #First recurrent layer with dropout
    model.add(Bidirectional(cudaLSTM(window_size, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_value))

    #Second recurrent layer with dropout
    model.add(Bidirectional(cudaLSTM((window_size*2), return_sequences=True)))
    model.add(Dropout(dropout_value))

    #Third recurrent layer
    model.add(Bidirectional(cudaLSTM(window_size, return_sequences=False)))

    #Output layer (returns the predicted value)
    model.add(Dense(units=1))
    
    #Set activation function
    model.add(Activation(activation_function))

    #Set loss function and optimizer
    model.compile(loss=loss_function, optimizer=optimizer,metrics=[rmseMetric])
    
    return model

def fit_model(model,X_train,Y_train,batch_num,num_epoch,val_split):
     #Record the time the model starts training
    start = time.time()

    #Train the model on X_train and Y_train
    history = model.fit(X_train, Y_train, batch_size= batch_num, nb_epoch=num_epoch, validation_split= val_split)
    plotTrainingInfo(history)

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

def plotTrainingInfo(history):
    # Plot training Details
    # Plot MSE values
    # print(history.history)
    # input("wait")
    plt.plot(history.history['rmseMetric'])
    plt.plot(history.history['val_rmseMetric'])

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
    plt.xlabel("Days")
    plt.ylabel('Feature Value')
    plt.legend(featureList,loc='upper left')
    plt.show()
    
    plt.plot(targets)
    plt.title('Target Values(Average Price of BTC)')
    plt.xlabel("Days")
    plt.ylabel('Target Value')
    plt.legend(targetList,loc='upper left')
    plt.show()

def plotTest(X,Y,target,legend):
    plt.plot(X)
    plt.plot(Y)
    plt.title("Test Prediction vs Target")
    plt.ylabel(target)
    plt.xlabel('Days')
    # legend.append("features: " + "".join(features))
    plt.legend(legend,loc='upper left')
    plt.show()

def loadData2(filename,window_length):
    BATCH_SIZE=64  
    EPOCHS = 70
    np.random.seed(0)
 
    raw_data = pd.read_csv(filename,header=0)    
 
    # features = ['Volume BTC','unnormalized speculation','Open','Close','velocity','delta vol']
    features = ['unnormalized speculation','velocity','delta vol','average price']
    targets = ['average price']

    training_X = raw_data[features]
    training_Y = raw_data[targets]


    training_X = training_X.values
    training_Y = training_Y.values
    training_X = np.flip(training_X,axis=0)
    training_Y = np.flip(training_Y,axis=0)
    print(training_Y)
    # Change target to be predicting tomorrows price
    # training_Y = np.roll(training_Y,1)
    training_Y = training_Y[1:]
    print(training_Y)
    listTrain = list(training_X)
    listTrain.pop()
    training_X = np.asarray(listTrain)

    # Shuffle the data
    # When data is shuffled training gets worse makes Sense!
    # np.random.shuffle(training_X)
    # np.random.shuffle(training_Y)
    # plotFeatures(training_X[:],training_Y[:],features,targets)
    
    split = int(len(training_X)*.9)
    # Data preprocess
    X_train = training_X[:split]
    X_test = training_X[split:]
    Y_train = training_Y[:split]
    Y_test = training_Y[split:]
    origYTest= Y_test

    sc = MinMaxScaler() #Normalize to 1
    # sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    Y_train = sc.fit_transform(Y_train)
    X_test = sc.fit_transform(X_test)
    Y_test = sc.fit_transform(Y_test)

    # X_train[:]
    # plotFeatures(X_train[:],Y_train[:],features,targets)

    X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
    Y_train = np.reshape(Y_train,(Y_train.shape[0],Y_train.shape[1]))
    X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
    Y_test = np.reshape(Y_test,(Y_test.shape[0],Y_test.shape[1]))

    regressor = init_model(window_length, 0.2, 'tanh', 'mse', 'rmsprop',X_train)
    regressor.summary()

    print('Training')
   
    # Fitting the RNN to the Training set

    regressor, trainingtime = fit_model(regressor,X_train,Y_train,BATCH_SIZE,EPOCHS,0.05)

    print('Evaluation')
    loss, mse = regressor.evaluate([X_test], Y_test,
                            batch_size=BATCH_SIZE)
    print('Test loss / test mse = {:.4f} / {:.4f}'.format(loss, mse))

    # make prediction
    testPredict = regressor.predict(X_test)
    trainPredict = regressor.predict(X_train)

    # calculate root mean squared error
    trainScore = rmse(trainPredict,Y_train)
    print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = rmse(sc.inverse_transform(testPredict),origY_test)

    testScore = rmse(testPredict,Y_test)
    print('Test Score: %.2f RMSE' % (testScore))

    print("Unnormalized Version")

    trainScore = rmse(sc.inverse_transform(trainPredict),sc.inverse_transform(Y_train))
    print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = rmse(sc.inverse_transform(testPredict),origY_test)

    testScore = rmse(sc.inverse_transform(testPredict),origYTest)
    print('Test Score: %.2f RMSE' % (testScore))

    
    s = "features-{}".format(",".join(features))

    plotTest(Y_test,testPredict,"average price",["Y_test","Y Prediction",s])
    y_predict, real_y_test, real_y_predict, fig1 = test_model(regressor, X_test, Y_test, origYTest)
    #Show the plot
    plt.show(fig1)

    
    return regressor,testScore


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def rmseMetric (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true)))
# def rmseMetric(y_true,y_pred):
#     return np.sqrt((np.mean((y_pred-y_true)**2)))

def main():

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

