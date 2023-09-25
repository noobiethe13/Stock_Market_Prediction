import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Reading stock market data from CSV file
def read_intraday_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Data preprocessing
def preprocess_data(data):
    target = data['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(target)

    return scaled_data, scaler

# Creation of the neural network model
def create_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Training of the neural network model
def train_model(model, x_train, y_train, epochs):
    model.fit(x_train, y_train, epochs=epochs, verbose=0)

# Prediction using the trained model
def predict_stock_prices(model, x_test, scaler):
    y_pred = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(y_pred)
    return predicted_prices

# Percentage accuracy calculation
def calculate_accuracy(actual_prices, predicted_prices):
    accuracy = (1 - np.abs(actual_prices - predicted_prices) / actual_prices) * 100
    return accuracy

# Plotting of the actual and predicted stock prices
def plot_stock_prices(actual_prices, predicted_prices):
    plt.plot(actual_prices, label='Actual')
    plt.plot(predicted_prices, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

def main():
    data = read_intraday_data("intraday_data.csv")

    scaled_data, scaler = preprocess_data(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    x_train = train_data[:-1]
    y_train = train_data[1:]
    x_test = test_data[:-1]
    y_test = test_data[1:]

    model = create_model()
    train_model(model, x_train, y_train, epochs=100)

    predicted_prices = predict_stock_prices(model, x_test, scaler)

    actual_prices = scaler.inverse_transform(y_test)

    plot_stock_prices(actual_prices, predicted_prices)

    accuracy = calculate_accuracy(actual_prices, predicted_prices)
    avg_accuracy = np.mean(accuracy)
    print(f'Average Percentage Accuracy: {avg_accuracy:.2f}%')

if __name__ == '__main__':
    main()