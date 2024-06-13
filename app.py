import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
import math
from sklearn.metrics import mean_squared_error

# Function to create 1D data into time-series
def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset) - step_size - 1):
        a = dataset[i:(i + step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)

# Load and preprocess data
@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M', dayfirst=False, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.reindex(index=df.index[::-1])
    return df

# Plot function
def plot_series(time, series, title):
    fig, ax = plt.subplots()
    ax.plot(time, series)
    ax.set_title(title)
    st.pyplot(fig)

# Streamlit app
st.title("Bitcoin Google Trends Prediction App")

option = st.selectbox(
    'Select Data to Visualize',
    ('Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume Traded')
)

# Load dataset
df = load_data("Cleaned Data/Bitcoin1D.csv")

if option == 'Open Price':
    data = df['PriceOpen']
elif option == 'High Price':
    data = df['PriceHigh']
elif option == 'Low Price':
    data = df['PriceLow']
elif option == 'Close Price':
    data = df['PriceClose']
else:
    data = df['VolumeTraded']

# Calculate and display highest and lowest values
highest_value = data.max()
lowest_value = data.min()
st.write(f'Highest {option}: {highest_value}')
st.write(f'Lowest {option}: {lowest_value}')

time = np.arange(1, len(df) + 1)

# Plot selected data
plot_series(time, data, f'Bitcoin {option}')

# Prepare data for LSTM model
OHCL_avg = df[['PriceOpen', 'PriceHigh', 'PriceLow', 'PriceClose']].mean(axis=1)
OHCL_avg = np.reshape(OHCL_avg.values, (len(OHCL_avg), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
OHCL_avg = scaler.fit_transform(OHCL_avg)

train_OHLC = int(len(OHCL_avg) * 0.56)
test_OHLC = len(OHCL_avg) - train_OHLC
train_OHLC, test_OHLC = OHCL_avg[0:train_OHLC, :], OHCL_avg[train_OHLC:len(OHCL_avg), :]

trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY = new_dataset(test_OHLC, 1)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# Build and train LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(1, step_size)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=25, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

st.write(f"Train RMSE: {trainScore:.2f}")
st.write(f"Test RMSE: {testScore:.2f}")

trainPredictPlot = np.empty_like(OHCL_avg)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size,:] = trainPredict

testPredictPlot = np.empty_like(OHCL_avg)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHCL_avg)-1,:] = testPredict

OHCL_avg = scaler.inverse_transform(OHCL_avg)

fig, ax = plt.subplots()
ax.plot(OHCL_avg, 'g', label='Original Dataset')
ax.plot(trainPredictPlot, 'r', label='Training Set')
ax.plot(testPredictPlot, 'b', label='Predicted price/test set')
ax.set_title("Hourly Bitcoin Predicted Prices")
ax.set_xlabel('Hourly Time')
ax.set_ylabel('Close Price')
ax.legend()
st.pyplot(fig)
