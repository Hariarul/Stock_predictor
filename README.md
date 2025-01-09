# Stock Price Prediction
This project predicts stock prices using a deep learning model and visualizes the predictions with Streamlit.

## Features

Stock Data Retrieval: Fetches historical stock data from Yahoo Finance.

Prediction: Uses a trained deep learning model to predict future stock prices.

Visualization: Displays original stock prices along with moving averages (100, 200, and 250 days) and predictions.

## Installation

## Prerequisites

Required libraries listed in requirements.txt

pip install -r requirements.txt

## Model File
This project requires the pre-trained model Latest_stock_price_model.h5. Make sure to have it in the project directory.

## Usage

Install dependencies and make sure the model file is available.

## Run the Streamlit app:

streamlit run stock_pred.py

Enter a stock ID (e.g., TSLA for Tesla) to see the stock data, moving averages, and predictions.

## Code Overview

Stock Data: Retrieves historical stock data using yfinance.

Model: Loads a pre-trained model (Latest_stock_price_model.h5) for predictions.

Preprocessing: Scales stock data to [0,1] range using MinMaxScaler.

Prediction: The model predicts stock prices based on historical data.

Visualization: Displays graphs of original stock prices, moving averages, and predictions using Matplotlib and Streamlit.
