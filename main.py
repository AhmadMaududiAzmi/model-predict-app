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
from flask import Flask, jsonify, json, abort, request, send_file, g
from db import Database
from config import DevelopmentConfig as devconf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

import threading
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

# ==============================================[ Other - Start ]
# ==============================================[ Other - End ]

# ==============================================[ Routes - Start ]
# Melakukan train model dengan menggunakan data train
@app.route(f"{route_prefix}/traindata", methods=["GET"])
def trainData():
    try:
        # nm_komoditas = request.args.get('komoditas_id', '')
        # nm_pasar = request.args.get('pasar_id', '')
        # tanggal_awal = request.args.get('start_date', '')
        # tanggal_akhir = request.args.get('end_date', '')

        # query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}' GROUP BY tanggal"
        # records = db.run_query(query=query)
        # db.close_connection()

        nm_komoditas = 9
        nm_pasar = 10
        tanggal_awal = '2016-01-01'
        tanggal_akhir = '2020-12-31'

        query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}' GROUP BY tanggal"
        records = db.run_query(query=query)
        db.close_connection()

        # Get data dan parsing menjadi time series
        data = pd.DataFrame(records, columns=['tanggal', 'harga_current'])

        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data.set_index('tanggal', inplace=True)
        # dataframe_json = data.to_json(orient='records')

        # Preprocessing data
        # Perhitungan rata-rata untuk mengisi data harga_current = 0
        avg_harga_current = data['harga_current'].mean()
        data['harga_current'] = data['harga_current'].replace(0, avg_harga_current)
        dataframe_json = data.reset_index().to_json(orient='records')
        # Normalisasi data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['harga_current']])
        data_json = json.dumps(data_scaled.tolist())

        # Pembagian data dengan membuat rumus len of percentage (data train dan data test)
        PERCENTAGE = 0.8 # Persentase data train adalah 0.8 (80%) untuk saat ini
        train_size = int(len(data_scaled) * PERCENTAGE)
        
        # Prepare data train
        data_train = data_scaled[:train_size]
        jumlah_data_train = f"Jumlah data = {data_train.shape}"

        # Pembentukan sequences data / data time series dari data train untuk model prediksi
        SEQUENCE_DATA = 60
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
        rmse = np.sqrt(np.mean(predictions - yTest) ** 2)
        akurasi = f"Root Mean Squared Error (RMSE) pada data test: {str(rmse)}"

        # Pembuatan dataframe setelah dilakukan pelatihan model
        predicted_data = pd.DataFrame({'Predicted': predictions.flatten(), 'Actual': yTest_original.flatten()}, index=data.index[-len(predictions):])
        predicted_data_json = predicted_data.to_json()

        # Simpan model LSTM ke dalam file sehingga saat penggunaan model saat prediksi tidak perlu train lagi
        name_model = f"model_{nm_komoditas}_{nm_pasar}_{tanggal_awal}_{tanggal_akhir}.joblib"
        joblib.dump(model, name_model, compress=False)

        save_to_database(nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir, name_model, predicted_data_json)

        response = {
            'predicted_data': predicted_data.to_json(),
            'model': name_model
        }

        # return jsonify(response)
        return predicted_data.to_json()
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Save output to database
def save_to_database(nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir, name_model, predicted_data_json):
    try:
        query = "INSERT INTO pertanian.hasil_prediksi (komoditas_id, pasar_id, tanggal_awal, tanggal_akhir, model, predicted_data) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir, name_model, predicted_data_json)
        db.run_query(query=query, values=values)
        db.close_connection()
        return "Successfully saved to database"
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Load model from database
@app.route(f"{route_prefix}/loadmodel", methods=['GET'])
def load_model_from_database():
    try:
        a = [1, 2, 3]
        return a
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Fungsi prediksi menggunakan LSTM
@app.route(f"{route_prefix}/predictdata", methods=['GET'])
def predictdata():
    try:
        # Mengambil parameter dari request
        komoditas_id = request.args.get('komoditas_id')
        pasar_id = request.args.get('pasar_id')
        tanggal_awal = request.args.get('start_date')
        tanggal_akhir = request.args.get('end_date')
        new_predicted_data = int(request.args.get('new_predicted_data'))  # Jumlah hari ke depan untuk prediksi

        # Mendapatkan model dari database berdasarkan parameter
        query_model = """
            SELECT model
            FROM hasil_prediksi
            WHERE komoditas_id = %s AND pasar_id = %s AND tanggal_awal = %s AND tanggal_akhir = %s
        """
        values_model = (komoditas_id, pasar_id, tanggal_awal, tanggal_akhir)
        result_model = db.run_query(query=query_model, values=values_model)
        model_path = result_model[0]['model']

        # Memuat model dari file
        model = joblib.load(name_model)

        # Mengambil data untuk prediksi
        query_data = f"""
            SELECT tanggal, harga_current
            FROM daftar_harga
            WHERE komoditas_id = %s AND pasar_id = %s AND tanggal BETWEEN %s AND %s
            ORDER BY tanggal DESC
        """
        values_data = (komoditas_id, pasar_id, tanggal_awal, tanggal_akhir)
        result_data = db.run_query(query=query_data, values=values_data)
        data = pd.DataFrame(result_data, columns=['tanggal', 'harga_current'])
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data.set_index('tanggal', inplace=True)

        # Normalisasi data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['harga_current']])

        # Persiapkan data untuk prediksi
        SEQUENCE_DATA = 60
        x_predict = []

        for i in range(SEQUENCE_DATA, len(data_scaled)):
            x_predict.append(data_scaled[i - SEQUENCE_DATA:i, 0])

        x_predict = np.array(x_predict)
        x_predict_3d = np.reshape(x_predict, (x_predict.shape[0], SEQUENCE_DATA, 1))

        # Lakukan prediksi
        new_predictions = model.predict(x_predict_3d)
        new_predictions = scaler.inverse_transform(new_predictions)

        # Persiapkan hasil prediksi untuk response
        new_predicted_data = pd.DataFrame({'Predicted': new_predictions.flatten()}, index=data.index[-len(new_predictions):])

        return new_predicted_data.to_json()
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
