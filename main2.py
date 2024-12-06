import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import io # test
import base64 # test
from io import BytesIO # test
import pymysql
import matplotlib # test
matplotlib.use('Agg') # test
import matplotlib.pyplot as plt
import joblib

from http import HTTPStatus
from flask_cors import CORS
from flask import Flask, jsonify, json, abort, request, send_file, session
from db import Database
from config import DevelopmentConfig as devconf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

import threading
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.metrics import RootMeanSquaredError
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

host = os.environ.get('FLASK_SERVER_HOST', devconf.HOST)
port = os.environ.get('FLASK_SERVER_PORT', devconf.PORT)
version = str(devconf.VERSION).lower()
url_prefix = str(devconf.URL_PREFIX).lower()
route_prefix = f"/{url_prefix}/{version}"

def create_app():
    app = Flask(__name__)
    cors = CORS(app, resources={f"{route_prefix}/*": {"origins": "http://localhost:8000"}})
    app.config.from_object(devconf)
    return app

def get_response_msg(data, status_code):
    message = {
        'status': status_code,
        'data': data if data else 'Record tidak ditemukan'
    }
    response_msg = jsonify(message)
    response_msg.status_code = status_code
    return response_msg

app = create_app()
wsgi_app = app.wsgi_app
db = Database(devconf)

# ==============================================[ Variables - Start ]
PERCENTAGE_TRAIN = 0.8
SEQUENCE_LENGTH = 7
LAYERS = 3
NEURONS_1 = 60
NEURONS_2 = 60
NEURONS_3 = 60
DROPOUT_RATE = 0.4
LEARN_RATE = 0.0001
BATCH_SIZE = 32
EPOCH = 100
# ==============================================[ Variables - End ]

# ==============================================[ Function - Start ]
def save_to_database(nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir, name_model, test_result):
    try:
        full_path = os.path.abspath(name_model)
        query = "INSERT INTO pertanian.hasil_prediksi (komoditas_id, pasar_id, tanggal_awal, tanggal_akhir, model, test_result) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir, full_path, test_result)
        db.run_query(query=query, values=values)
        db.close_connection()
        return "Successfully saved to database"
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

def update_database(komoditas_id, pasar_id, tanggal_awal, tanggal_akhir, name_model, test_result):
    try:
        full_path = os.path.abspath(name_model)
        query = "UPDATE hasil_prediksi SET test_result = (%s) WHERE komoditas_id = (%s) AND pasar_id = (%s) AND tanggal_awal = (%s) AND tanggal_akhir = (%s) AND model = (%s)"
        values = (test_result, komoditas_id, pasar_id, tanggal_awal, tanggal_akhir, full_path)
        db.run_query(query=query, values=values)
        db.close_connection()
        return "Database updated"
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Data sequence
def create_dataset(data, n_sequence):
    try:
        X, y = [], []
        for i in range(len(data) - n_sequence):
            X.append(data[i:i + n_sequence])
            y.append(data[i + n_sequence])
        return np.array(X), np.array(y)
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))
# ==============================================[ Function - End ]

# ==============================================[ Routes - Start ]
@app.route(f"{route_prefix}/train", methods=["GET"])
def train():
    try:
        nm_komoditas = 8
        nm_pasar = 1
        tanggal_awal = '2016-01-01'
        tanggal_akhir = '2020-12-31'

        query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}'"
        records = db.run_query(query=query)
        db.close_connection()

        data = pd.DataFrame(records, columns=['tanggal', 'harga_current'])
        
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data.set_index('tanggal', inplace=True)
        # dataframe_json = data.to_json(orient='records')

        avg_harga_current = data['harga_current'].mean()
        data['harga_current'] = data['harga_current'].replace(0, avg_harga_current)
        # dataframe_json = data.reset_index().to_json(orient='records')
        # dataframe_json = data.to_json(orient='records')

        Q1 = data['harga_current'].quantile(0.25)
        Q3 = data['harga_current'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data['harga_current'][(data['harga_current'] < (Q1 - 1.5 * IQR)) | (data['harga_current'] > (Q3 + 1.5 * IQR))]
        median_harga = data['harga_current'].median()
        data['harga_current'][outliers.index] = median_harga
        dataframe_json = data.to_json()

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data['harga_current'].values.reshape(-1, 1))
        dataframe_scaled = pd.DataFrame(data_scaled, columns=['harga_scaled'], index=data.index)

        name_scaler = f"data_scaled_{nm_komoditas}_{nm_pasar}_{tanggal_awal}_{tanggal_akhir}.pkl"
        joblib.dump(scaler, name_scaler)
        dataframe_json = dataframe_scaled.to_json()

        train_size = int(len(data_scaled) * PERCENTAGE_TRAIN)
        data_train = data_scaled[0:train_size, :]
        print("data train: ", len(data_train))

        xTrain, yTrain = create_dataset(data_train, SEQUENCE_LENGTH)

        xTrain_3d = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
        data_xTrain_3d = f"Bentuk data array 3 dimensi xTrain = {xTrain_3d.shape}"

        model = Sequential()
        model.add(LSTM(60, return_sequences=True, input_shape=(xTrain_3d.shape[1], 1)))
        model.add(Dropout(0.4))
        model.add(LSTM(60, return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(60))
        model.add(Dense(25))
        model.add(Dense(1))
        # model.compile(optimizer='adam', loss='mean_squared_error')
        model.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.001), metrics= [RootMeanSquaredError()])
        
        # model.fit(xTrain_3d, yTrain, epochs=100, batch_size=32)
        model.fit(xTrain_3d, yTrain, epochs=10, batch_size=32)
        # history = model.fit(xTrain_3d, yTrain, validation_data=(xVal_3d, yVal), epochs=50, batch_size=32)

        name_model = f"model_{nm_komoditas}_{nm_pasar}_{tanggal_awal}_{tanggal_akhir}.h5"
        model.save(name_model)

        # Save test data to database
        a = "NULL"

        # plt.figure(figsize=(12, 6))
        # plt.plot(history.history['loss'], 'b-', label='Training Loss')
        # plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
        # plt.title('Training and Validation Loss Curves')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.grid(True)

        # img = io.BytesIO()
        # plt.savefig(img, format='png')
        # img.seek(0)
        # plt.close()

        save_to_database(nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir, name_model, a)

        # return send_file(img, mimetype='image/png')
        return "Model Trained"
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

@app.route(f"{route_prefix}/test", methods=["GET"])
def test():
    try:
        nm_komoditas = 8
        nm_pasar = 1
        test_tanggal_awal = '2016-01-01'
        test_tanggal_akhir = '2020-12-31'
        tanggal_awal = '2016-01-01' # get model
        tanggal_akhir = '2020-12-31' # get model 

        query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{test_tanggal_awal}' AND tanggal <= '{test_tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}'"
        records = db.run_query(query=query)
        db.close_connection()

        data = pd.DataFrame(records, columns=['tanggal', 'harga_current'])
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data.set_index('tanggal', inplace=True)

        avg_harga_current = data['harga_current'].mean()
        data['harga_current'] = data['harga_current'].replace(0, avg_harga_current)

        Q1 = data['harga_current'].quantile(0.25)
        Q3 = data['harga_current'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data['harga_current'][(data['harga_current'] < (Q1 - 1.5 * IQR)) | (data['harga_current'] > (Q3 + 1.5 * IQR))]
        median_harga = data['harga_current'].median()
        data['harga_current'][outliers.index] = median_harga

        scaler = MinMaxScaler(feature_range=(0,1))
        data_scaled = scaler.fit_transform(data['harga_current'].values.reshape(-1, 1))
        dataframe_scaled = pd.DataFrame(data_scaled, columns=['harga_scaled'], index=data.index)
        
        train_size = int(len(data_scaled) * PERCENTAGE_TRAIN)
        data_test = data_scaled[train_size:, :]
        print("data test: ", len(data_test))

        xTest, yTest = create_dataset(data_test, SEQUENCE_LENGTH)

        xTest_3d = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
        data_xTest_3d = f"Bentuk data array 3 dimensi xTest = {xTest_3d.shape}"
        
        # name_model = f"model_8_5_2016-01-01_2020-12-31.h5"
        name_model = f"model_{nm_komoditas}_{nm_pasar}_{tanggal_awal}_{tanggal_akhir}.h5"
        model = load_model(name_model)

        predictions = model.predict(xTest_3d)

        actual_values = scaler.inverse_transform(yTest.reshape(-1,1))
        predictions_values = scaler.inverse_transform(predictions.reshape(-1,1))

        # predictions_list = predictions_values.tolist()
        # predictions_json = json.dumps(predictions_list)

        results = pd.DataFrame({
            'tanggal': data.index[train_size + SEQUENCE_LENGTH:],
            'harga_actual': actual_values.flatten(),
            'harga_predicted': predictions_values.flatten()
        })

        update_database(nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir, name_model, predictions_values)

        # Evaluasi
        # mse = mean_squared_error(actual_values, predictions_values)
        rmse = np.sqrt((np.mean(actual_values - predictions_values) ** 2))
        # rmse = np.sqrt(mse)
        # mape = mean_absolute_percentage_error(actual_values, predictions_values)
        # deviation = np.mean(actual_values - predictions_values)
        # akurasi = f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}, Deviasi: {deviation:.2f}"
        akurasi = f"RMSE: {rmse:.2f}"

        plt.figure(figsize=(14, 7))
        plt.plot(results['tanggal'], results['harga_actual'], label='Actual Prices')
        plt.plot(results['tanggal'], results['harga_predicted'], label='LSTM Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(akurasi)
        plt.legend()
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')
        # return jsonify(predictions_json)
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Fungsi prediksi menggunakan LSTM
@app.route(f"{route_prefix}/predict", methods=['GET'])
def predict():
    try:
        nm_komoditas = 8
        nm_pasar = 1
        tanggal_awal = '2016-01-01'
        tanggal_akhir = '2020-12-31' # get model
        new_tanggal_akhir = '2020-12-31'
        n_days = 61

        query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{new_tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}'"
        records = db.run_query(query=query)
        db.close_connection()
     
        data = pd.DataFrame(records, columns=['tanggal', 'harga_current'])
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data.set_index('tanggal', inplace=True)

        avg_harga_current = data['harga_current'].mean()
        data['harga_current'] = data['harga_current'].replace(0, avg_harga_current)

        Q1 = data['harga_current'].quantile(0.25)
        Q3 = data['harga_current'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data['harga_current'][(data['harga_current'] < (Q1 - 1.5 * IQR)) | (data['harga_current'] > (Q3 + 1.5 * IQR))]
        median_harga = data['harga_current'].median()
        data['harga_current'][outliers.index] = median_harga

        scaler = MinMaxScaler(feature_range=(0,1))
        data_scaled = scaler.fit_transform(data['harga_current'].values.reshape(-1, 1))

        name_model = f"model_{nm_komoditas}_{nm_pasar}_{tanggal_awal}_{tanggal_akhir}.h5"
        model = load_model(name_model)

        # query_test_result = f"SELECT test_result FROM hasil_prediksi WHERE komoditas_id = {nm_komoditas} AND pasar_id = {nm_pasar} AND tanggal_awal = {tanggal_awal} AND tanggal_akhir = {tanggal_akhir}"
        # records_test_result = db.run_query(query=query_test_result)
        # db.close_connection()

        # test_result = pd.DataFrame(records_test_result)

        predictions = []
        last_seq = data_scaled[-SEQUENCE_LENGTH:]
        for _ in range(n_days):
            pred_input = np.reshape(last_seq, (1, SEQUENCE_LENGTH, 1))
            pred = model.predict(pred_input)
            predictions.append(pred[0][0])

            last_seq = np.append(last_seq[1:], [[pred[0][0]]], axis=0)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        future_dates = pd.date_range(start=new_tanggal_akhir, periods=n_days + 1, freq='D')[1:]
        results = pd.DataFrame({
            'tanggal': future_dates,
            'harga_predicted': predicted_prices.flatten()
        })

        plt.figure(figsize=(14, 7))
        # plt.plot(data.index, data['harga_current'], label='Actual Prices', color='blue')
        # plt.plot(data.index, test_result, label='Prediction Price', color='orange')
        plt.plot(results['tanggal'], results['harga_predicted'], label='Predicted Prices', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f"Prediksi Harga {n_days} Hari Ke Depan untuk Komoditas {nm_komoditas} di Pasar {nm_pasar}")
        plt.legend()
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))
# =================================================[ Routes - End ]

# ================================[ Error Handler Defined - Start ]
# HTTP 404 error handler
@app.errorhandler(HTTPStatus.NOT_FOUND)
def page_not_found(e):
    return get_response_msg(data=str(e), status_code=HTTPStatus.NOT_FOUND)


# HTTP 400 error handler
@app.errorhandler(HTTPStatus.BAD_REQUEST)
def bad_request(e):
    return get_response_msg(str(e), HTTPStatus.BAD_REQUEST)


# HTTP 500 error handler
@app.errorhandler(HTTPStatus.INTERNAL_SERVER_ERROR)
def internal_server_error(e):
    return get_response_msg(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)
# ==================================[ Error Handler Defined - End ]


if __name__ == '__main__':
    # Launch the application
    app.run(host=host, port=port)
