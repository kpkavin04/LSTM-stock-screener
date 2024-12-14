import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import nest_asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

def fetch_stock_data(ticker, period="2y"):
    #to download stock data, multi-threading allowed
    stock_data = yf.download(ticker, period=period, threads=True)  
    stock_data.index = pd.to_datetime(stock_data.index, utc=True)  
    return stock_data

def preprocess_data(stock_data):
    #extract the closing values
    data = stock_data[['Close']].values
    #closing values are scaled between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(time_series_data, sequence_length=60):
    x = []
    y = []
    #iterate through the closing price of the stock
    for i in range(sequence_length, len(time_series_data)):
        x.append(time_series_data[i-sequence_length:i, 0])
        y.append(time_series_data[i, 0])
    return np.array(x), np.array(y)

def build_and_train_lstm_model(x_train, y_train, epochs=50, batch_size=32):
    #init
    model = Sequential()
    #first lstm layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    #second lstm layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    #third lstm layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    #dense layer
    model.add(Dense(units=1))
    
    #adaptive moment estimation is usde as an optimizer as it is a good default
    #huber loss function is used so as to avoid penalizing large fluctuations
    model.compile(optimizer='adam', loss='huber')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Define the input schema for the API
class StockRequest(BaseModel):
    ticker: str

# API route to predict stock prices
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/predict")
def predict_stock_prices(request: StockRequest):
    ticker = request.ticker

    # Fetch stock data
    stock_data = fetch_stock_data(ticker)
    if stock_data.empty:
        raise HTTPException(status_code=404, detail="Stock data not found for the provided ticker.")

    # Preprocess data
    scaled_data, scaler = preprocess_data(stock_data)
    x_train, y_train = create_sequences(scaled_data)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Train or load the model
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        model = build_and_train_lstm_model(x_train, y_train)
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)

    # Prepare the input for prediction
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

    # Predict the next 30 days
    predictions = []
    rng = np.random.default_rng(seed = 0)
    for _ in range(30):
        predicted_price = model.predict(last_60_days)

        #slight noise introduced to avoid a cascading effect on the predictions
        noise = rng.normal(loc=0, scale=0.000001)  
        predicted_price[0, 0] += noise

        predictions.append(predicted_price[0, 0])
        #since the last_60_days is a 3d array, we need to reshape the 2d predicted price
        predicted_price = np.array(predicted_price).reshape(1, 1, 1) 
        #the new prediction is appended to the last 60 days so as to be used for the next prediction
        last_60_days = np.append(last_60_days[:, 1:, :], predicted_price, axis=1)

    # Inverse transform the predictions
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    return {"ticker": ticker, "predictions": predicted_prices.tolist()}

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)