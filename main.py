import os
import pymysql
from http import HTTPStatus
from flask_cors import CORS
from flask import Flask, redirect, jsonify, json, request,url_for, abort
from db import Database
from config import DevelopmentConfig as devconf

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
        'status' : status_code,
        'data' : data if data else 'Record tidak ditemukan'
    }
    response_msg = jsonify(message)
    response_msg.status_code = status_code
    return response_msg

app = create_app()
wsgi_app = app.wsgi_app
db = Database(devconf)

## ==============================================[ Routes - Start ]
# Get data From Database
@app.route(f"{route_prefix}/getcomodities", methods=["GET"])
def getData():
    try:
        # comoditiescode = request.args.get('komoditas', default=None, type=str)
        query = f"SELECT tanggal, nm_pasar, nm_komoditas, harga_current FROM pertanian.harga_komoditas WHERE nm_komoditas = 'BAWANG PUTIH' AND nm_pasar = 'Pasar Blimbing' GROUP BY tanggal"
        records = db.run_query(query=query)
        response = get_response_msg(records, HTTPStatus.OK)
        db.close_connection()
        return response
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))


# Preprocessing data
@app.route(f"{route_prefix}/preprosesdata", methods=['GET', 'POST'])
def processData():
    try:
        query = f"SELECT tanggal, nm_pasar, nm_komoditas, harga_current FROM pertanian.harga_komoditas WHERE nm_komoditas = 'BAWANG PUTIH' AND nm_pasar = 'Pasar Blimbing' AND harga_current = '0' GROUP BY tanggal"
        records = db.run_query(query=query)
        response = get_response_msg(records, HTTPStatus.OK)    
        # Mencari missing values pada harga_current lalu dihapus atau dihitung rata-ratanya untuk diisi ke dalam missing values
        # Melakukan proses normalisasi dataframe yang sudah difilter sesuai inputan user
        # Membagi data ke dalam data train dan data test lalu simpan ke dalam 2 dataframe (sesuai persenan yg diinputkan user)
        db.close_connection()
        return response
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Train data train
@app.route(f"{route_prefix}/traindata", methods=['GET', 'POST'])
def trainData():
    return 

# Proses Prediksi
@app.route(f"{route_prefix}/prediksi", methods=['GET', 'POST'])
def prediksi():
    try:
        print
        # Mengambil data train untuk pelatian model
        # Membuat model prediksi
        # Melakukan training pada data train (sesuai jumlah persenan yg diinputkan user)
        # Melakukan training pada data testing
        # Save data to json and visualize it to graph (plot graph)
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Testing for connecting to database
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
## =================================================[ Routes - End ]

## ================================[ Error Handler Defined - Start ]
## HTTP 404 error handler
@app.errorhandler(HTTPStatus.NOT_FOUND)
def page_not_found(e):    
    return get_response_msg(data=str(e), status_code=HTTPStatus.NOT_FOUND)


## HTTP 400 error handler
@app.errorhandler(HTTPStatus.BAD_REQUEST)
def bad_request(e):
    return get_response_msg(str(e), HTTPStatus.BAD_REQUEST)


## HTTP 500 error handler
@app.errorhandler(HTTPStatus.INTERNAL_SERVER_ERROR)
def internal_server_error(e):
    return get_response_msg(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)
## ==================================[ Error Handler Defined - End ]

if __name__ == '__main__':
    ## Launch the application 
    app.run(host=host, port=port)