from flask import Flask, render_template, request, redirect
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



# Loading Flask Application
app = Flask(__name__)


# Home route
@app.route('/')
def index():
    """
    Displays the home page with a form to enter stock ticker
    """
    return render_template('index.html')


# Predict route
@app.route('/predict', methods=["POST"])
def predict():
    """
    Displays the predicted share price, summary and plot for a stock.
    Expects a form input named stock_ticker via post method
    """

    # Get the input stock ticker
    stock_ticker = request.form.get("stock_ticker")

    # Validate the stock_ticker
    stock_ticker = stock_ticker.strip().upper()
    stock = yf.Ticker(stock_ticker)

    # If not a valid stock ticker, redirect to home route
    if stock.info["regularMarketPrice"] == None:
        return redirect("/")

    # Get business summary
    summary = stock.info["longBusinessSummary"]

    # Get prediction value and the image string
    prediction, image = predict_next_value(stock_ticker)

    return render_template("predict.html", stock_ticker=stock_ticker, summary=summary, prediction=prediction, image=image)



def predict_next_value(tickerSymbol):
    """
    Predicts the next day share price for a stock.
    Expects a tickerSymbol which is the ticker for the stock.
    Returns the predicted share price and a .png image encoded as a base64 string.
    """

    """
    # loading data
    START = dt.datetime(2015, 1, 1)
    END = dt.datetime.now()
    main_data = web.DataReader(tickerSymbol, 'yahoo', START, END)
    main_data.reset_index(inplace=True)
    """

    # Data
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime(2021, 6, 1)
    data = web.DataReader(tickerSymbol, 'yahoo', start, end)
    data = data.reset_index()

    # predicition day
    pred_day = 60

    # Scalar
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Test Model
    model = load_model("model.h5");

    test_start = dt.datetime(2021, 6, 1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(tickerSymbol, 'yahoo', test_start, test_end)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - pred_day:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Prediction of test data
    x_test = []
    for x in range(pred_day, len(model_inputs)+1):
        x_test.append(model_inputs[x-pred_day:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)


    # Graph (Final Predicted by ML Model with Actual Data set) 
    """
    This is code written by Nilesh, I replaced this code with the code written
    below for the plot.


    plt.figure(figsize=(12,6))
    plt.plot(actual_prices, color="black", label=f"Actual {tickerSymbol} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {tickerSymbol} Price")
    plt.title(f"{tickerSymbol} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{tickerSymbol} Share Price')
    plt.legend()
    """

    fig = plt.figure(figsize=(10,5))
    axes = fig.add_axes([0.1,0.1,0.6,0.6])
    axes.plot(actual_prices, color="black", label=f"Actual {tickerSymbol} Price")
    axes.plot(predicted_prices, color="green", label=f"Predicted {tickerSymbol} Price")
    axes.set_title(f"{tickerSymbol} Share Price")
    axes.set_xlabel("Time")
    axes.set_ylabel(f"{tickerSymbol} Share Price")
    axes.legend()
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    #Next day
    real_data = [model_inputs[len(model_inputs+1) - pred_day : len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction) # final Ans

    return (prediction[0][0], pngImageB64String)


# Main Function
if __name__ == '__main__':
    app.run(debug=True)
