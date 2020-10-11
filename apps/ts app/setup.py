import numpy as np
import tensorflow as tf
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Conv1D

def plot_series(time, series, format="-", start=0, end=None):
    """plot the time series

    Args:
        time (list/np.array): the time data
        series (list/np.array): the data
        format (str, optional): the dash line type. Defaults to "-".
        start (int, optional): from when to plot. Defaults to 0.
        end ([type], optional): till when to plot. Defaults to None.
    """  
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
    If window_size=1, then this is equivalent to naive forecast
    Args:
    series (list/np.array): the data
    window_size (int): the window size

    Returns:
    [np.array]: the predictions
    """  
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

def create_dataset(df):
    """create a train/test scaled dataset

    Args:
        df (pandas DataFrame): the data

    Returns:
        [5 objects]: train/test input and output data and a scaler object
    """    
    df['output'] = df['Adj Close'].shift(-1)
    df = df.dropna()

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    rescaled = scaler.fit_transform(df.values)

    training_ratio = 0.8
    training_testing_index = int(len(rescaled) * training_ratio)
    training_data = rescaled[:training_testing_index]
    testing_data = rescaled[training_testing_index:]
    training_length = len(training_data)
    testing_length = len(testing_data)
    # Split training into input/output. Output is the one we added to the end
    training_input_data = training_data[:, 0:-1]
    training_output_data = training_data[:, -1]

    # Split testing into input/output. Output is the one we added to the end
    testing_input_data = testing_data[:, 0:-1]
    testing_output_data = testing_data[:, -1]
    training_input_data = training_input_data.reshape(training_input_data.shape[0], 1, training_input_data.shape[1])
    testing_input_data = testing_input_data.reshape(testing_input_data.shape[0], 1, testing_input_data.shape[1])
    return training_input_data, training_output_data, testing_input_data, testing_output_data, scaler

def unscaled_predict(model, scaler, testing_input_data, testing_output_data):
    """predict real values with unscaling

    Args:
        model (tf model): tf model to predict
        scaler (scaler object): the scaler to do unscale of the data
        testing_input_data (list): the testing input data
        testing_output_data (list): the testing output data

    Returns:
        [2 objects]: unscaled actual data and unscaled predictions
    """    
    raw_predictions = model.predict(testing_input_data)

    testing_input_data = testing_input_data.reshape((testing_input_data.shape[0], testing_input_data.shape[2]))
    testing_output_data = testing_output_data.reshape((len(testing_output_data), 1))

    # Invert scaling for prediction data
    unscaled_predictions = concatenate((testing_input_data, raw_predictions), axis = 1)
    unscaled_predictions = scaler.inverse_transform(unscaled_predictions)
    unscaled_predictions = unscaled_predictions[:, -1]

    # Invert scaling for actual data
    unscaled_actual_data = concatenate((testing_input_data, testing_output_data), axis = 1)
    unscaled_actual_data = scaler.inverse_transform(unscaled_actual_data)
    unscaled_actual_data = unscaled_actual_data[:, -1]
    return unscaled_actual_data, unscaled_predictions

def moving_average(df):
    """preparing the data into the moving average function

    Args:
        df (pandas DataFrame): the data

    Returns:
        [2 objects]: the original values and the moving average prediction
    """    
    values = df['Close']
    split_time = 100
    windoq_size = 30
    moving_avg = moving_average_forecast(values, windoq_size)[split_time-windoq_size:]
    return values.iloc[split_time:], moving_avg

def LSTM_model(df):
    """create an LSTM model

    Args:
        df (pandas DataFrame): the data

    Returns:
        [3 objects]: unscaled actual data, unscaled predictions, history
    """    
    training_input_data, training_output_data, testing_input_data, testing_output_data, scaler = create_dataset(df)
    model = Sequential()
    model.add(LSTM(100, input_shape = (training_input_data.shape[1], training_input_data.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss='mse')

    history = model.fit(
            training_input_data,
            training_output_data,
            epochs = 50,
            validation_data=(testing_input_data, testing_output_data),
            shuffle=False,
            verbose=0
            )
    unscaled_actual_data, unscaled_predictions = unscaled_predict(model, scaler, testing_input_data, testing_output_data)
    return unscaled_actual_data, unscaled_predictions, history

def BiLSTM_model(df):
    """create an BiLSTM model

    Args:
        df (pandas DataFrame): the data

    Returns:
        [3 objects]: unscaled actual data, unscaled predictions, history
    """    
    training_input_data, training_output_data, testing_input_data, testing_output_data, scaler = create_dataset(df)

    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(training_input_data.shape[1], training_input_data.shape[2])))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss='mse')

    history = model.fit(
            training_input_data,
            training_output_data,
            epochs = 50,
            validation_data=(testing_input_data, testing_output_data),
            shuffle=False,
            verbose=0
            )
    unscaled_actual_data, unscaled_predictions = unscaled_predict(model, scaler, testing_input_data, testing_output_data)
    return unscaled_actual_data, unscaled_predictions, history

def CNN_LSTM_model(df):
    """create an CNN_LSTM model

    Args:
        df (pandas DataFrame): the data

    Returns:
        [3 objects]: unscaled actual data, unscaled predictions, history
    """ 
    training_input_data, training_output_data, testing_input_data, testing_output_data, scaler = create_dataset(df)

    model = Sequential()
    model.add(Conv1D(filters=60, kernel_size=5,
                        strides=1, padding="causal",
                        activation="relu",
                        input_shape=(training_input_data.shape[1], training_input_data.shape[2])))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss='mse')

    history = model.fit(
            training_input_data,
            training_output_data,
            epochs = 50,
            validation_data=(testing_input_data, testing_output_data),
            shuffle=False,
            verbose=0
            )
    unscaled_actual_data, unscaled_predictions = unscaled_predict(model, scaler, testing_input_data, testing_output_data)
    return unscaled_actual_data, unscaled_predictions, history