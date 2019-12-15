#import all Relevant Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#Import auto_arima from Arima
#Install arima using command 'pip install pmdarima'
from pmdarima.arima import auto_arima

#Read Data File
data = pd.read_csv(r'D:\All Assignment\Statistics\XRP_DATA.csv')

#To check data
data.head()


#Check all rows and Column
data.shape


#Train algorithm and set model
train= data[:900]
valid = data[900:]

training = train['ClosingPrice']
validation = valid['ClosingPrice']


model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)

#forecasting values for 178 days
forecast = model.predict(n_periods=178)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#calculate RMS
rms=np.sqrt(np.mean(np.power((np.array(valid['ClosingPrice'])-np.array(forecast['Prediction'])),2)))
rms

#plot
plt.plot(train['ClosingPrice'])
plt.plot(valid['ClosingPrice'])
plt.plot(forecast['Prediction'])
