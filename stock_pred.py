import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime
import streamlit as st
from tensorflow.keras.metrics import MeanSquaredError

st.title("Stock price prediction")

stock = st.text_input("Enter the stock ID","TSLA")

end = datetime.now()

start = datetime(end.year-5,end.month,end.day)

df = yf.download(stock,start, end,multi_level_index=False)

model = load_model("Latest_stock_price_model.h5", compile=False)


st.subheader("Stock data")

st.write(df)

split = int(len(df)*0.7)
x_test = pd.DataFrame(df.Close[split:])

def plot_graph(fig_size,values,full_data,extra_data=0,extra_dataset=None):
    fig = plt.figure(figsize=fig_size)
    plt.plot(values,'g')
    plt.plot(full_data.Close,'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader("Original closing price and MA for 250 days")
df['MA_for_250_days'] = df.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6),df['MA_for_250_days'],df,0))

st.subheader("Original closing price and MA for 200 days")
df['MA_for_200_days'] = df.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6),df['MA_for_200_days'],df,0))

st.subheader("Original closing price and MA for 100 days")
df['MA_for_100_days'] = df.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6),df['MA_for_100_days'],df,0))


st.subheader("Original closing price and MA for 100 days and MA for 250 days")
st.pyplot(plot_graph((15,6),df['MA_for_100_days'],df,1,df['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_data = scaler.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    { 'Original_test_data':inv_y_data.reshape(-1),
     'predictions':inv_pre.reshape(-1)
    },
    index = df.index[split+100:])
st.subheader("Original values vs Predicted values")
st.write(plotting_data)

st.subheader("Original close price vs Predicted close price")
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([df.Close[:split+100],plotting_data],axis=0))
plt.legend(["Data not used","Original test data","predicted_test_data"])
st.pyplot(fig)