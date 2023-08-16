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
        

        # Get data from input user
        nm_komoditas = request.args.get('komoditas_id', '')
        nm_pasar = request.args.get('pasar_id', '')
        tanggal_awal = request.args.get('start_date', '')
        tanggal_akhir = request.args.get('end_date', '')

        query = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{tanggal_akhir}' AND komoditas_id = '{nm_komoditas}' AND pasar_id = '{nm_pasar}' GROUP BY tanggal"
        records = db.run_query(query=query)
        db.close_connection()
        #return jsonify(records)
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

        # Data harga yang sudah diolah
        # dataset = pd.DataFrame(data_scaled, columns=['harga_current'], index=data.index).reset_index()

        # Pembagian data dengan membuat rumus len of percentage (data train dan data test)
        PERCENTAGE = 0.8 # Persentase data train adalah 0.8 (80%) untuk saat ini
        train_size = int(len(data_scaled) * PERCENTAGE)
        
        # Prepare data train
        # return 1
        data_train = data_scaled[:train_size]
        data_train_json = data_train.tolist()
        jumlah_data_train = f"Jumlah data = {data_train.shape}"

        # Pembentukan sequences data / data time series dari data train untuk model prediksi
        SEQUENCE_DATA = 30 # Menggunakan 30 data sebagai inputan model LSTM. Sehingga tidak keseluruhan data train menjadi 1 inputan ke dalam model LSTM
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

        # Mengembalikan values data test ke bentuk asal sebelum normalisasi
        yTest_original = scaler.inverse_transform(yTest.reshape(-1, 1))

        # Evaluasi model menggunakan RMSE
        mse = mean_squared_error(yTest, predictions)
        rmse = np.sqrt(mse)
        akurasi = "Root Mean Squared Error (RMSE) pada data test: {:.2f}".format(rmse)

        # Pembuatan dataframe setelah dilakukan pelatihan model
        # return 2
        data_predictions = pd.DataFrame(predictions.flatten(), columns=['predictions'], index=data.index[-len(predictions):])
        data_predictions_json = data_predictions.to_dict(orient='records')
        # return 3
        data_valid = pd.DataFrame(yTest_original.flatten(), columns=['data_test_unscaled'], index=data.index[-len(predictions):])
        data_valid_json = data_valid.to_dict(orient='records')
        # Reset index kolom
        # data_predictions.reset_index(drop=True, inplace=True)
        # data_valid.reset_index(drop=True, inplace=True)

        # Penggabungan data asli dengan data prediksi
        dataset = pd.concat([data_valid, data_predictions], axis=1)
        # return 4
        dataset_json = dataset.to_dict(orient='records')

        # Plot data (data_train, data_valid(red: data_test), data_predict) menjadi grafik
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(dataset.index, dataset['data_test_unscaled'], label='Actual')
        # ax.plot(dataset.index, dataset['predictions'], label='Predicted')
        # ax.plot(data.index, data['harga_current'], label='Train Data', linestyle='dashed', alpha=0.5)
        # ax.set_title('Actual vs. Predicted Prices\nRMSE: ' + str(rmse))
        # ax.set_xlabel('Years')
        # ax.set_ylabel('Prices')
        # ax.legend()

        # Simpan gambar pada direktori sebagai PNG
        # return 5
        plot_filename = generate_plot_filename(nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir)
        # plt.savefig(plot_filename, format='png')
        # plt.close()
        #grap = plotGrap(dataset, plot_filename, data)
        thread = threading.Thread(target=plotGrap,args=(dataset, plot_filename, data))
        thread.start()
        arr = {
            'filename' : plot_filename,
            'data_predictions' : data_predictions_json,
            'data_valid' : data_valid_json,
            'data_train' : data_train_json
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
