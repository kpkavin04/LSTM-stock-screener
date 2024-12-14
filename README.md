# LSTM-stock-screener

This project is a machine learning-based API that **predicts stock prices for the next 30 days** using historical data. It makes use of Long Short-Term Memory (LSTM) neural networks to model time-series data and offers predictions via **FastAPI**.

## Features
- Fetches historical stock data using Yahoo Finance.
- Preprocesses data with Min-Max scaling for optimal model performance.
- Utilizes a multi-layer LSTM model.
- Introduces slight random noise to predictions to enhance robustness and reduce cascading effects.
- Exposes predictions through a **FastAPI** application, featuring CORS support for frontend integration.

## How It Works
- Fetches and preprocesses historical stock data for the specified ticker.
- Trains an LSTM model using the closing price data.
- Generates 30-day predictions iteratively, incorporating random noise for variability.
- Outputs predictions in human-readable format via the API, displaying predictions in the form of a line plot.
