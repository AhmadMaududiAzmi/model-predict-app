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

# ==============================================[ Function - Start ]
# Plot Graph (3 parameter; data_train, data_valid, data_prediction)
def plotGrap(dataset, plot_filename, data):
    plt.switch_backend('agg')
    plt.figure(figsize=(12, 6))
    plt.plot(dataset.index, dataset['data_test_unscaled'], label='Actual Prices')
    plt.plot(dataset.index, dataset['predictions'], label='Predicted Prices')
    plt.plot(data.index, data['harga_current'], label='Train Data', color='green')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Years')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig(plot_filename, format='png')
    plt.close()
    return plot_filename

# Generate filename grafik sesuai data
def generate_plot_filename(nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir):
    filename = f"plot_{nm_komoditas.replace(' ', '_')}_{nm_pasar.replace(' ', '_')}_{tanggal_awal.replace(' ', '_')}_{tanggal_akhir.replace(' ', '_')}.png"
    return filename
# ==============================================[ Function - End ]

# ==============================================[ Routes - Start ]
# Melakukan train model dengan menggunakan data train
SEQUENCE_DATA = 60
@app.route(f"{route_prefix}/traindata", methods=["GET", "POST"])
def trainData():
    try:
        query = "SELECT tanggal, harga_current FROM pertanian.daftar_harga WHERE tanggal >= '2016-01-01' AND tanggal <= '2020-12-31' AND komoditas_id = '8' AND pasar_id = '8' GROUP BY tanggal"
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

        nm_komoditas = request.args.get('komoditas_id', '')
        nm_pasar = request.args.get('pasar_id', '')
        tanggal_awal = request.args.get('start_date', '')
        tanggal_akhir = request.args.get('end_date', '')
        
        #query = "SELECT tanggal, harga_current FROM pertanian_lama.harga_komoditas WHERE tanggal >= '2016-01-01' AND tanggal <= '2020-12-31' AND nm_komoditas = 'BAWANG PUTIH' AND nm_pasar = 'Pasar Dinoyo' GROUP BY tanggal"
        query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}' GROUP BY tanggal"
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
        rmse = np.sqrt(np.mean(predictions - yTest) ** 2)
        akurasi = f"Root Mean Squared Error (RMSE) pada data test: {str(rmse)}"

        # Pembuatan dataframe setelah dilakukan pelatihan model
        predicted_data = pd.DataFrame({'Predicted': predictions.flatten(), 'Actual': yTest_original.flatten()}, index=data.index[-len(predictions):])
        
        # Plot hasil prediksi dan data asli
        plt.switch_backend('agg')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(predicted_data.index, predicted_data['Actual'], label='Actual')
        ax.plot(predicted_data.index, predicted_data['Predicted'], label='Predicted')
        ax.plot(data.index, data['harga_current'], label='Train Data', linestyle='dashed', alpha=0.5)
        ax.set_title('Hasil Prediksi vs. Data Asli\nRMSE: ' + str(rmse))
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga')
        ax.legend()

        # Simpan gambar sebagai file PNG
        plot_filename = generate_plot_filename(nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir)
        plt.savefig(plot_filename, format='png')
        plt.close()

        # Mengirimkan gambar ke browser
        #return send_file(plot_filename, mimetype='image/png')

        arr = {
            'filename' : plot_filename
        }
        return jsonify(arr)
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))
    # root.mainloop()
#sendfile
@app.route(f"{route_prefix}/get_chart", methods=['GET'])
def get_chart():
    name = request.args.get('name', '')
    return send_file(name, mimetype='image/png')

#testarray
@app.route(f"{route_prefix}/testarray", methods=['GET'])
def testarray():
    name = request.args.get('name', '')
    array = {
        'data':name,
        'value':'0'

    }
    return jsonify(array)

# Add new predict data
@app.route(f"{route_prefix}/addPredict", methods=['GET'])
# Penggunaan parameter untuk menambahkan variabel data baru yang berupa hasil prediksi masa mendatang
def addPredict():
    try:
        # Load model LSTM
        model = load_model('trained_model.h5')

        # Get data terakhir dari data
        last_date_data = data.index(-1)
        last_value_data = data['harga_current'][-1]

        # Filter data
        new_query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}' GROUP BY tanggal"
        record = db.run_query(query=new_query)
        db.close_connection()

        # Data baru untuk prediksi masa mendatang
        new_data = []
        for i in range(next_predict):
            new_data.append(last_value_data)
            last_value_data = model.predict(new_data[-SEQUENCE_DATA:].reshape(1, SEQUENCE_DATA, 1))
            new_data.append(last_value_data[0][0])

        new_data = np.array(new_data).reshape(-1, 1)
        new_data_scaled = scaler.transform(new_data)

        xNew = []
        for i in range(SEQUENCE_DATA, len(new_data_scaled)):
            xNew.append(new_data_scaled[i-SEQUENCE_DATA:i, 0])
        
        xNew = np.array(xNew)
        xNew_3d = np.reshape(xNew, (xNew.shape[0], SEQUENCE_DATA, 1))

        # Melakukan prediksi pada data tambahan
        predictions_new = model.predict(xNew_3d)
        predictions_new = scaler.inverse_transform(predictions_new)

        # Membuat indeks tanggal baru untuk data tambahan
        date_range = pd.date_range(start=last_data_date + pd.DateOffset(days=1), periods=len(predictions_new))

        # Membuat DataFrame untuk data tambahan
        data_new = pd.DataFrame(predictions_new, columns=['predictions'], index=date_range)

        # Menggabungkan data tambahan dengan data asli
        combined_data = pd.concat([data, data_new])

        # Simpan gambar pada direktori sebagai PNG
        plot_filename_new = generate_plot_filename(nm_komoditas, nm_pasar, last_data_date, date_range[-1])
        thread_new = threading.Thread(target=plotGrap, args=(combined_data, plot_filename_new, data))
        thread_new.start()

        arr_new = {
            'filename': plot_filename_new,
            'data_combined': combined_data.to_dict(orient='records')
        }

        return jsonify(arr_new)
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
