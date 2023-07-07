import pandas as pd
import numpy as np
import datetime
import os
import pymysql
from http import HTTPStatus
from flask_cors import CORS
from flask import Flask, redirect, jsonify, json, request, url_for, abort
from db import Database
from config import DevelopmentConfig as devconf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


host = os.environ.get('FLASK_SERVER_HOST', devconf.HOST)
port = os.environ.get('FLASK_SERVER_PORT', devconf.PORT)
version = str(devconf.VERSION).lower()
url_prefix = str(devconf.URL_PREFIX).lower()
route_prefix = f"/{url_prefix}/{version}"

def create_app():
    app = Flask(__name__)
    cors = CORS(app, resources={f"{route_prefix}/*": {"origins": "*"}})
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
        
        # Normalisasi menggunakan MinMaxScaler()
        scaler = MinMaxScaler()
        data['harga_scaled'] = scaler.fit_transform(data[['harga_current']])

        # Split data ke data train dan data test
        X = data.drop('harga_current', axis=1)
        y = data['harga_current']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Fungsi untuk melakukan train data
@app.route(f"{route_prefix}/traindata", methods=['GET', 'POST'])
def trainData():
    try:
        if request.method == 'GET':
            return
        if request.method == 'POST':
            print('POST')
            return
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Fungsi prediksi menggunakan LSTM
@app.route(f"{route_prefix}/prediksi", methods=['GET', 'POST'])
def prediksi():
    try:
        # Mengambil data train untuk pelatian model
        # Membuat model prediksi
        # Melakukan training pada data train (sesuai jumlah persenan yg diinputkan user)
        # Melakukan training pada data testing
        # Save data to json and visualize it to graph (plot graph)
        a = 'halo'
        return a
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Testing for connecting to database (/api/v1/health)
@app.route(f"{route_prefix}/health", methods=['GET'])
def health():
    try:
        db_status = "Tersambung ke DB" if db.db_connection_status else "Tidak dapat tersambung ke DB"
        response = get_response_msg("Anda masuk " + db_status, HTTPStatus.OK)
        return response
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
