import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
from io import BytesIO
import pymysql
import matplotlib.pyplot as plt
import joblib

from http import HTTPStatus
from flask_cors import CORS
from flask import Flask, redirect, jsonify, json, url_for, abort, request, send_file
from db import Database
from config import DevelopmentConfig as devconf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

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

# =================================================[ Routes - End ]
# Variabel-variabel
SEQUENCE_DATA = 30
NEXT_PREDICTION = 30

# Plot grafik hasil prediksi
@app.route(f"{route_prefix}/plot", methods=['GET'])
def plot():
    try:
        query = "SELECT tanggal, harga_current FROM pertanian.daftar_harga WHERE tanggal >= '2016-01-01' AND tanggal <= '2020-12-31' AND nm_komoditas = 'Bawang Merah' AND nm_pasar = 'Pasar Wlingi' GROUP BY tanggal"
        records = db.run_query(query=query)
        db.close_connection()

        # Get data dan parsing menjadi time series
        data = pd.DataFrame(records, columns=['tanggal', 'harga_current'])
        dateParse = lambda x: pd.to_datetime(x)
        data['tanggal'] = data['tanggal'].apply(dateParse)
        data = data.sort_values('tanggal')
        data.set_index('tanggal', inplace=True)

        # Preprocessing data
        # Perhitungan rata-rata untuk mengisi data harga_current = 0
        avg_harga_current = data['harga_current'].mean()
        data['harga_current'] = data['harga_current'].replace(0, avg_harga_current)
        # Normalisasi data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['harga_current']])
        data_json = json.dumps(data_scaled.tolist())

        # Dataframe harga yang sudah diolah
        dataset = pd.DataFrame(data_scaled, columns=['harga_current'], index=data.index).reset_index()

        # Pembagian data dengan membuat rumus len of percentage (data train dan data test)
        PERCENTAGE = 0.8 # Persentase data train adalah 0.8 (80%) untuk saat ini
        train_size = int(len(data_scaled) * PERCENTAGE)
        
        # Prepare data train
        data_train = data_scaled[:train_size]
        jumlah_data_train = f"Jumlah data = {data_train.shape}"

        # # Pembentukan sequences data / data time series dari data train untuk model prediksi
        xTrain = []
        yTrain = []
        for i in range(SEQUENCE_DATA, len(data_train)):
            xTrain.append(data_train[i-SEQUENCE_DATA:i, 0])
            yTrain.append(data_train[i, 0])

        # Convert trained x dan y sebagai numpy array
        xTrain, yTrain = np.array(xTrain), np.array(yTrain)
        jumlah_xTrain = f"Jumlah data =  {len(yTrain)}"

        # Bentuk ulang x trained data menjadi array 3 dimension
        xTrain_3d = np.reshape(xTrain, (xTrain.shape[0], SEQUENCE_DATA, 1))
        data_xTrain_3d = f"Bentuk data array 3 dimensi xTrain = {xTrain_3d.shape}"

        # Model LSTM
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(xTrain_3d.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Train data
        model.fit(xTrain_3d, yTrain, epochs=100, batch_size=32)

        # Prepare data test
        data_test = data_scaled[train_size - SEQUENCE_DATA:]
        jumlah_data_test = f"Jumlah data = {data_test.shape}"

        # Pembentukan sequences data / data time series dari data test untuk model prediksi        
        xTest = []
        yTest = data_scaled[train_size:]
        for i in range(SEQUENCE_DATA, len(data_test)):
            xTest.append(data_test[i - SEQUENCE_DATA:i, 0])
        
        # Convert tested x dan y sebagai numpy array
        xTest, yTest = np.array(xTest), np.array(yTest)

        # Bentuk ulang x tested data menjadi array 3 dimension
        xTest_3d = np.reshape(xTest, (xTest.shape[0], SEQUENCE_DATA, 1))
        data_xTest_3d = f"Bentuk data array 3 dimensi xTrain = {xTest_3d.shape}"

        # Melakukan prediksi pada data test
        predictions = model.predict(xTest_3d)
        # Mengembalikan values data ke bentuk asal sebelum dinormalisasi
        predictions = scaler.inverse_transform(predictions) 

        # Mengembalikan nilai asli data test sebelum dinormalisasi (red: data_valid)
        yTest_original = scaler.inverse_transform(yTest.reshape(-1, 1))

        # Evaluasi model menggunakan RMSE
        mse = mean_squared_error(yTest, predictions)
        # rmse = np.sqrt(mse)
        rmse = np.sqrt(np.mean(predictions - yTest) ** 2)
        akurasi = f"Root Mean Squared Error (RMSE) pada data test: {str(rmse)}"

        # Pembuatan dataframe setelah dilakukan pelatihan model
        predicted_data = pd.DataFrame({'Predicted': predictions.flatten(), 'Actual': yTest_original.flatten()}, index=data.index[-len(predictions):])
        
        # Plot hasil prediksi dan data asli
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(predicted_data.index, predicted_data['Actual'], label='Actual')
        ax.plot(predicted_data.index, predicted_data['Predicted'], label='Predicted')
        ax.plot(data.index, data['harga_current'], label='Train Data', linestyle='dashed', alpha=0.5)
        ax.set_title('Hasil Prediksi vs. Data Asli\nRMSE: ' + str(rmse))
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga')
        ax.legend()

        # Simpan gambar sebagai file PNG
        plot_filename = 'plot.png'
        plt.savefig(plot_filename, format='png')
        plt.close()

        # Mengirimkan gambar ke browser
        return send_file(plot_filename, mimetype='image/png')
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Add new predict data
@app.route(f"{route_prefix}/addPredict", methods=['GET'])
# Penggunaan parameter untuk menambahkan variabel data baru yang berupa hasil prediksi masa mendatang
def addPredict(next_predict = 30):
    try:
        # Pengambilan variabel dengan global pada def predict()

        # Dataset baru harga komoditas
        new_dataset = pd.DataFrame(data_scaled, columns=['harga_current'], data=data.index).reset_index()
         
        # Prepare data train         
        new_data_train = dataset[:train_size] # Data train
        new_data_test = dataset[train_size:] # Data test

        # Pembentukan sequences data / data time series dari data test untuk model prediksi
        new_xTest = []
        new_yTest = new_dataset[train_size:]
        for i in range(SEQUENCE_DATA, len(new_data_test)):
            new_xTest.append(new_data_test[i - SEQUENCE_DATA:i, 0])

        # Convert tested x dan y sebagai numpy array
        new_xTest, new_yTest = np.array(new_xTest), np.array(new_yTest)

        # Bentuk ulang x trained data menjadi array 3 dimension
        new_xTest_3d = np.reshape(new_xTest, (new_xTest.shape[0], SEQUENCE_DATA, 1))

        # Melakukan prediksi pada data test baru
        new_predictions = model.predict(new_xTest)
        new_predictions = scaler.inverse_transform(new_predictions)

        new_yTest_original = scaler.inverse_transform(new_yTest.reshape(-1, 1))

        # Evaluasi model menggunakan RMSE
        new_mse = mean_squared_error(new_yTest, new_predictions)
        # rmse = np.sqrt(mse)
        new_rmse = np.sqrt(np.mean(new_predictions - new_yTest) ** 2)
        akurasi = f"Root Mean Squared Error (RMSE) pada data test: {str(new_rmse)}"

        # Pembuatan dataframe setelah dilakukan pelatihan model
        new_predicted_data = pd.DataFrame(new_predictions, columns=['predictions'])

        # Pembuatan variabel pembanding antara data_train, data_valid, dan data_predictions yang baru

        return
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
