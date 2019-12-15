#importing libraries
import pandas as pd
import numpy as np

#reading the data
df = pd.read_csv(r'D:\All Assignment\Statistics\working file\ETH_DATA.csv')

# looking at the first five rows of the data
df.head()
df.shape

# setting the index as date
df['Date'] = pd.to_datetime(df.Date)
df.index = df['Date']

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'ClosingPrice'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['ClosingPrice'][i] = data['ClosingPrice'][i]

# splitting into train and validation
train = new_data[:900]
valid = new_data[900:]

# shapes of training set
train.shape

# shapes of validation set
valid.shape

# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# making predictions
preds = []
for i in range(0,valid.shape[0]):
    a = train['ClosingPrice'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)

rms=np.sqrt(np.mean(np.power((np.array(valid['ClosingPrice'])-preds),2)))
rms

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['ClosingPrice'])
plt.plot(valid[['ClosingPrice', 'Predictions']])