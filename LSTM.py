#Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
%matplotlib inline

#Read data into CSV
data = pd.read_csv(r'D:\All Assignment\Statistics\XRP_DATA.csv')

#check data by selecting top five rows
data.head()


#creating dataframe
new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'ClosingPrice'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['ClosingPrice'][i] = data['ClosingPrice'][i]

#creating index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:900,:]
valid = dataset[900:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 178 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


#Calculate rms value
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms


#Plot the graph
train = new_data[:900]
valid = new_data[900:]
valid['Predictions'] = closing_price
plt.plot(train['ClosingPrice'])
plt.plot(valid[['ClosingPrice','Predictions']])