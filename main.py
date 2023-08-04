import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import io
import pymysql
import matplotlib as plt
import joblib

from http import HTTPStatus
from flask_cors import CORS
from flask import Flask, redirect, jsonify, json, url_for, abort, render_template, Response
from db import Database
from config import DevelopmentConfig as devconf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

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

# Evaluasi model prediksi
def evaluate_model(model, X_test, y_test, y_pred):
    # Menghitung Mape
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Menghitung RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Menghitung MSE
    mse = mean_squared_error(y_test, y_pred)

    return mape, rmse, mse

app = create_app()
wsgi_app = app.wsgi_app
db = Database(devconf)

# ==============================================[ Routes - Start ]
# Get data dari database
@app.route(f"{route_prefix}/getcomodities", methods=["GET"])
def getData():
    try:
        # query = "SELECT * FROM pertanian.harga_komoditas WHERE nm_komoditas = 'Gula Pasir Dalam Negri' AND nm_pasar = 'Pasar Blimbing' GROUP BY tanggal"
        query = "SELECT * FROM pertanian.harga_komoditas WHERE nm_komoditas = 'Gula Pasir Dalam Negri'"
        records = db.run_query(query=query)
        response = get_response_msg(records, HTTPStatus.OK)
        db.close_connection()
        return response
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Fungsi untuk proses pengolahan data; missing values, scaler, normalization
@app.route(f"{route_prefix}/getdataframe", methods=["GET"])
def processData():
    try:
        query = "SELECT * FROM pertanian.harga_komoditas WHERE nm_komoditas = 'BAWANG PUTIH' AND nm_pasar = 'Pasar Dinoyo'"
        records = db.run_query(query=query)
        db.close_connection()
        data = pd.DataFrame(records, columns=['id', 'tanggal', 'nm_pasar', 'nm_komoditas', 'id_komuditas', 'harga_current'])
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data = data.sort_values('tanggal')

        # Mengabaikan values harga_current = 0
        # data = data[data['harga_current'] != 0]

        # Menghitung rata-rata tiap bulan untuk mengisi values harga_current = 0
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data['bulan'] = data['tanggal'].dt.month
        monthly_avg = data.groupby('bulan')['harga_current'].mean()
        data.loc[data['harga_current'] == 0, 'harga_current'] = data['bulan'].map(monthly_avg) 
        
        # Split data ke data train dan data test
        X = data.drop('harga_current', axis=1)
        y = data['harga_current']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisasi menggunakan MinMaxScaler()
        scaler = MinMaxScaler()
        data['harga_scaled'] = scaler.fit_transform(data[['harga_current']])

        return process_function(X_train, X_test)
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

def process_function(data_train, data_test):
    processed_data = {
        'data_train': data_train.to_dict(orient='records'),
        'data_test': data_test.to_dict(orient='records')
    }
    return jsonify(processed_data)

# Sintaks asli disimpan di simpan.py. Saat ini sebagai percobaan
# Melakukan train model dengan menggunakan data train
@app.route(f"{route_prefix}/traindata", methods=["GET", "POST"])
def trainData():
    try:
        query = "SELECT * FROM pertanian.harga_komoditas WHERE nm_komoditas = 'BAWANG PUTIH' AND nm_pasar = 'Pasar Dinoyo'"
        records = db.run_query(query=query)
        db.close_connection()

        # Get dan parsing data menjadi time series
        data = pd.DataFrame(records, columns=['id', 'tanggal', 'nm_pasar', 'nm_komoditas', 'id_komuditas', 'harga_current'])
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data['tanggal_epoch'] = data['tanggal'].apply(lambda x: x.timestamp())
        data = data.sort_values('tanggal')

        # Menghitung rata-rata tiap bulan untuk mengisi values harga_current = 0
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data['bulan'] = data['tanggal'].dt.month
        monthly_avg = data.groupby('bulan')['harga_current'].mean()
        data.loc[data['harga_current'] == 0, 'harga_current'] = data['bulan'].map(monthly_avg) 
        
        # # Normalisasi menggunakan MinMaxScaler()
        scaler = MinMaxScaler()
        data['harga_scaled'] = scaler.fit_transform(data[['harga_current']])

        # Split data ke data train dan data test
        X = data['tanggal_epoch'].values.reshape(-1, 1)
        y = data['harga_scaled'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(50))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Train data
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        # Simpan model LSTM ke dalam file sehingga saat penggunaan model saat prediksi tidak perlu train lagi
        joblib.dump(model, 'trained_model.joblib')

        # Response
        # response = {
        #     'message': 'Model successfully trained'
        # }

        # return jsonify(response)
        return "Model trained successfuly"
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Fungsi prediksi menggunakan LSTM
@app.route(f"{route_prefix}/predict", methods=['GET'])
def predict():
    try:
        query = "SELECT tanggal, harga_current FROM pertanian.daftar_harga WHERE tanggal >= '2016-01-01' AND tanggal <= '2020-12-31' AND nm_komoditas = 'Bawang Merah' AND nm_pasar = 'Pasar Dinoyo' GROUP BY tanggal"
        records = db.run_query(query=query)
        db.close_connection()

        # Get data dan parsing menjadi time series
        dateParse = lambda x: pd.to_datetime(x)
        data = pd.DataFrame(records, columns=['tanggal', 'harga_current'])
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

        # Dataframe data yang sudah diolah
        df = pd.DataFrame(data_scaled, columns=['harga_current'], index=data.index).reset_index()
        
        # Tambahkan proses agar model get data start dan end timestamp sesuai inputan user

        # Pembagian data (data train dan data test)
        train_size = int(len(df) * 0.8) # Persentase data train adalah 0.8 (80%) untuk saat ini
        
        # Prepare data train
        data_train = df[:train_size]

        # Pembentukan sequences data / data time series dari data train untuk model prediksi
        SEQUENCE_DATA = 30 # Menggunakan 30 data sebagai inputan model LSTM. Sehingga tidak keseluruhan data train menjadi 1 inputan ke dalam model LSTM
        xTrain = []
        yTrain = []
        for i in range(SEQUENCE_DATA, len(data_train)):
            xTrain.append(data_train[i-SEQUENCE_DATA:i]['harga_current'].values)
            yTrain.append(data_train.iloc[i]['harga_current'])

        # Convert trained x dan y sebagai numpy array
        xTrain, yTrain = np.array(xTrain), np.array(yTrain)

        # Bentuk ulang x trained data menjadi array 3 dimension
        xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

        # Model LSTM
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train data
        model.fit(xTrain, yTrain, epochs=100, batch_size=32)

        # Save model yang sudah dilatih
        # model.save('model_trained.h5')

        # Panggil model yang sudah dilatih
        # loaded_model = load_model("model_trained.h5")

        # Prepare data test
        data_test = df[train_size:]

        # Pembentukan sequences data / data time series dari data test untuk model prediksi
        xTest = []
        yTest = []
        for i in range(SEQUENCE_DATA, len(data_test)):
            xTest.append(data_test.iloc[i - SEQUENCE_DATA:i]['harga_current'].values)
            yTest.append(data_test.iloc[i]['harga_current'])
        
        xTest = []
        yTest = df[train_size:, :]
        for i in range(SEQUENCE_DATA, len(data_test)):
            xTest.append(data_test[i - SEQUENCE_DATA:i]['harga_current'].values)
        
        # Convert tested x dan y sebagai numpy array
        xTest, yTest = np.array(xTest), np.array(yTest)

        # Bentuk ulang x tested data menjadi array 3 dimension
        xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

        # Melakukan prediksi pada data test
        # predictions = loaded_model.predict(xTest)
        predictions = model.predict(xTest)
        predictions = scaler.inverse_transform(predictions) # Mengembalikan values data ke bentuk asal sebelum dinormalisasi

        # Evaluasi model menggunakan RMSE
        mse = mean_squared_error(yTest, predictions)
        rmse = np.sqrt(mse)
        result = "Root Mean Squared Error (RMSE) pada data test: {:.2f}".format(rmse)

        # Pembuatan data train dan data valid untuk ditampilkan dalam grafik
        train = df.loc[:train_size, ['tanggal', 'harga_current']]
        valid = df.loc[train_size:, ['tanggal', 'harga_current']]

        # Pembuatan dataframe untuk hasil prediksi yang sudah dilakukan
        # dfPredictions = pd.DataFrame(predictions, columns=['predictions'], index=data.index).reset_index()
        predictions_dict = {'tanggal': data.index[SEQUENCE_DATA:].tolist(), 'predictions': predictions[:, 0].tolist()}

        # Pembuatan index 'tanggal'
        # dfPredictions = dfPredictions.reset_index()
        # valid = valid.reset_index()

        # Penggabungan data valid dengan data prediksi
        # valid = pd.concat([valid, dfPredictions], axis=1)

        # Print
        data_json = json.dumps(predictions_dict)

        return data_json
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))
        # traceback.print_exc()

# Add new predict data
@app.route(f"{route_prefix}/addPredict", methods=['GET'])
def addPredict():
    try:
        data_json = 123
        return jsonify(data_json)
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Testing for connecting to database (/api/v1/health)
@app.route(f"{route_prefix}/health", methods=['GET'])
def health():
    try:
        print("Hello")
        return "Halo tersampaikan"
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# /
@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('health'))
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
