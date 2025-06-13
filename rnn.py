#Recurrent Neural Networks

#Part 1 Data Preprocessing

#Installing necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the training sets

df= pd.read_csv("Google_Stock_Price_Train.csv")
training_set=df.iloc[:,1:2].values

#Feature Scaling

#Normalizing the values using min-max method
from sklearn.preprocessing import MinMaxScaler
fs=MinMaxScaler(feature_range=(0, 1)) #This is because in min-max scaler the values are always between 0 & 1
scaled_training_set=fs.fit_transform(training_set) #fit_transform not only fits the value but also transforms in the range of 0-1


#Creating data structure with 60 timesteps and 1 output
#60 seems to be the ideal time steps to avoid under fitting or overfitting of the model

X_train=[]
Y_train=[]

#The range starts from 60 and the training set is 1257 days, but in python loop upper bound is excluded, hence 1258

for r in range (60,1258):
    X_train.append(scaled_training_set[r-60:r,0])
    Y_train.append(scaled_training_set[r,0])                # x_train trains from 1-59 and y_train trains the 60th data, i.e. t+1 data

X_train,Y_train= np.array(X_train),np.array(Y_train)

#Reshaping
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #A 3D tensor or array, with shape (batch, timesteps, feature).

#Part 2 Building the RNN

#Importing the keras library and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
    
#Initializing the RNN

regressor=Sequential() #Regressor is used because we have to predict the continuous value, we use regression instesd of classification    

#Adding the first LSTM layer and some Dropout Regularization

regressor.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))

regressor.add(Dropout(0.2)) #20% of the neurons values will be dropped during the training to prevent overfitting

#Adding second, third and fourth LSTM layer and some Dropout Regularization

regressor.add(LSTM(50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(50,return_sequences=False)) #Since this is the last LSTM layer
regressor.add(Dropout(0.2))

#Adding the output layer

regressor.add(Dense(units=1)) #Final prediction has only one value

#Compiling the RNN
regressor.compile(optimizer="adam",loss='mean_squared_error')

#Fitting the RNN to the Training Set

regressor.fit(X_train,Y_train,epochs=200, batch_size=32)   #Y_train is the ground truth

#Part 3 Making the predictions and visualizing the results

#Getting the real stock price of 2017
df_test= pd.read_csv("Google_Stock_Price_Test.csv")
testing_set=df_test.iloc[:,1:2].values

#Getting the predicted stock price of 2017
dataset_total=pd.concat((df['Open'],df_test['Open']),axis=0) # Concateneting train and test sets
inputs=dataset_total[len(dataset_total)-len(df_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=fs.transform(inputs)

X_test=[]

for r in range (60,80):
    X_test.append(inputs[r-60:r,0])
    
X_test= np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price= regressor.predict(X_test)
predicted_stock_price=fs.inverse_transform(predicted_stock_price)   # Reverse scaling of the predicted price 


#Visulaizing the results
plt.plot(testing_set,color='red',label='Actual Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

        

