import json
import datetime
from setup import *
from flask_bootstrap import Bootstrap
from pandas_datareader import data as web
from flask import Flask, request, render_template

import random

app = Flask(__name__)
Bootstrap(app)
app.static_folder = 'static'

@app.route("/", methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    symbol = request.form["text"].upper()
    model = request.form.get("modelType")
    return target(symbol, model)
  else:
    return render_template('index.html')

def target(symbol, model):
	return render_template('loading.html', data = {"symbol": symbol, "model": model})

@app.route('/processing')
def processing():
    data = request.args.to_dict()
    model = data["model"][1:-1]
    symbol = data["symbol"][1:-1]
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365*8)).strftime("%m-%d-%Y")
    df = web.DataReader(symbol, data_source='yahoo', start=start_date)

    if model == "moovingAverage":
        unscaled_actual_data, unscaled_predictions = moving_average(df)
    elif model == "lstm":
        unscaled_actual_data, unscaled_predictions, history = LSTM_model(df)
    elif model == "bilstm":
        unscaled_actual_data, unscaled_predictions, history = BiLSTM_model(df)
    elif model == "cnn_lstm":
        unscaled_actual_data, unscaled_predictions, history = CNN_LSTM_model(df)
    else:
        return render_template('index.html')
    if model == "moovingAverage":
        y1 = [round(x, 4) for x in unscaled_actual_data]
        y2 = [round(x, 4) for x in unscaled_predictions]
        plot_data = {'x_axis': [list(range(len(unscaled_actual_data))), list(range(len(unscaled_predictions)))], 
                'y_axis': [y1, y2],
                'len':len(y1) + len(y2) - 1}
        return render_template("plotmoovingAverage.html",
                                plot_data=plot_data)

    y1_1 = [round(x, 4) for x in history.history['loss']]
    y2_1 = [round(x, 4) for x in history.history['val_loss']]
    y1_2 = [round(x, 4) for x in unscaled_actual_data]
    y2_2 = [round(x, 4) for x in unscaled_predictions]
    plot_data = {'x_axis1': [list(range(len(history.history['loss']))), list(range(len(history.history['val_loss'])))], 
                 'y_axis1': [y1_1, y2_1],
                 'x_axis2': [list(range(len(unscaled_actual_data))), list(range(len(unscaled_predictions)))], 
                 'y_axis2': [y1_2, y2_2],
                 'len1':len(y1_1) - 1, 
                 'len2':len(y1_2) - 1}
    # render_template_string(template,
    #                 plot_data=plot_data)
    return render_template("plot.html",
                                  plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)