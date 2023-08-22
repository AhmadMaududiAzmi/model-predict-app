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
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
PERCENTAGE = 0.8

# Plot grafik hasil prediksi
@app.route(f"{route_prefix}/plot", methods=['GET'])
def plot():
    try:
        # query = "SELECT tanggal, harga_current FROM pertanian_lama.harga_komoditas WHERE tanggal >= '2016-01-01' AND tanggal <= '2020-12-31' AND nm_komoditas = 'BAWANG MERAH' AND nm_pasar = 'Pasar Tawangmangu' GROUP BY tanggal"
        query = "SELECT tanggal, harga_current FROM pertanian.daftar_harga WHERE tanggal >= '2019-01-01' AND tanggal <= '2020-12-31' AND komoditas_id = '8' AND pasar_id = '102' GROUP BY tanggal"
        records = db.run_query(query=query)
        db.close_connection()

        # Get data dan parsing menjadi time series
        data = pd.DataFrame(records, columns=['tanggal', 'harga_current'])
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        # data = data.sort_values('tanggal')
        data.set_index('tanggal', inplace=True)
        # dataframe_json = data.to_json(orient='records')

        # Preprocessing data
        # Perhitungan rata-rata untuk mengisi data harga_current = 0
        avg_harga_current = data['harga_current'].mean()
        data['harga_current'] = data['harga_current'].replace(0, avg_harga_current)
        dataframe_json = data.reset_index().to_json(orient='records')
        # # Normalisasi data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['harga_current']])
        data_json = json.dumps(data_scaled.tolist())

        # Dataframe harga yang sudah diolah
        # dataset = pd.DataFrame(data_scaled, columns=['harga_current'], index=data.index).reset_index()

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
        def calculate_akurasi(predicted, actual):
            n = len(predicted)
            mse = np.sum((predicted - actual)**2) / n
            rmse = np.sqrt(mse)
            return rmse
        # mse = calculate_akurasi(predictions, yTest_original)
        rmse = calculate_akurasi(predictions, yTest_original)
        # mse = mean_squared_error(yTest_original, predictions)
        # rmse = np.sqrt(mse)
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
# @app.route(f"{route_prefix}/addPredict", methods=['GET'])
# # Penggunaan parameter untuk menambahkan variabel data baru yang berupa hasil prediksi masa mendatang
# def addPredict(data, nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir):
#     try:
#         # get last date data for creating new dataset
#         new_tanggal_akhir = tanggal_akhir + NEXT_PREDICTION
#         new_tanggal_akhir = pd.to_datetime()

#         # New query for getting new dataset
#         query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{new_tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}' GROUP BY tanggal"
#         records = db.run_query(query=query)
#         db.close_connection()

#         # Create new dataset
#         new_data = pd.DataFrame(records, columns=['tanggal', 'harga_current'])
#         dateParse = lambda x: pd.to_datetime(x)
#         new_data['tanggal'] = new_data['tanggal'].apply(dateParse)
#         new_data = new_data.sort_values('tanggal')
#         new_data.set_index('tanggal', inplace=True)

#         # Preprocessing data
#         # Perhitungan rata-rata untuk mengisi data harga_current = 0
#         avg_harga_current = new_data['harga_current'].mean()
#         new_data['harga_current'] = new_data['harga_current'].replace(0, avg_harga_current)
#         # Normalisasi data
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         new_data_scaled = scaler.fit_transform(new_data[['harga_current']])
#         data_json = json.dumps(new_data_scaled.tolist())

#         # Pembagian data dengan membuat rumus len of percentage (data train dan data test)
#         train_size = int(len(new_data_scaled) * PERCENTAGE)

#         # Create new data_test
#         new_data_test = new_data_scaled[train_size - SEQUENCE_DATA:]

#         # Create new sequence data_test for new dataset
#         new_xTest = []
#         new_yTest = new_data_scaled[train_size:]
#         for i in range(SEQUENCE_DATA, len(new_data_test)):
#             new_xTest.append(new_data_test[i - SEQUENCE_DATA:i, 0])
        
#         # Convert x and y to numpy
#         new_xTest = np.array(new_xTest)

#         # Convert to array 3 dimension
#         new_xTest_3d = np.reshape(new_xTest, (new_xTest.shape[0], SEQUENCE_DATA, 1))

#         # Load model prediction
#         model = load_model('trained_model.h5')

#         # Make new predict for future
#         new_predictions = model.predict(new_xTest)

#         # Transform to real values
#         new_predictions = scaler.inverse_transform(new_predictions)

#         # RMSE
#         new_rmse = np.sqrt(np.mean(new_predictions - new_yTest) ** 2)
#         print('Root mean square (RMSE) - New:' + str(new_rmse))
#         akurasi = "Root Mean Squared Error (RMSE) pada data test: {:.2f}".format(new_rmse)

#         # Create new dataframe for new_data_predictions, new_data_valid, new_data_train
#         new_data_train = new_data.loc[train_size:]
#         new_data_valid = new_data.loc[:train_size]
#         new_data_predictions = pd.DataFrame(new_predictions, columns=['new_predictions'], index=new_data.index[-len(new_predictions):])

#         # Concat
#         new_dataset = pd.concat([new_data_valid, new_data_predictions], axis=1)
#         return
#     except pymysql.MySQLError as err:
#         abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
#     except Exception as e:
#         abort(HTTPStatus.BAD_REQUEST, description=str(e))

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
