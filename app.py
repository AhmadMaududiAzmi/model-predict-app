@app.route(f"{route_prefix}/predictdata", methods=['GET'])
def predictdata():
    try:
        # Mengambil parameter dari request
        komoditas_id = request.args.get('komoditas_id')
        pasar_id = request.args.get('pasar_id')
        tanggal_awal = request.args.get('start_date')
        tanggal_akhir = request.args.get('end_date')
        new_predicted_data = int(request.args.get('new_predicted_data'))  # Jumlah hari ke depan untuk prediksi

        # Mengambil model dari database berdasarkan parameter
        query_model = f"SELECT model FROM hasil_prediksi WHERE komoditas_id = {komoditas_id} AND pasar_id = {pasar_id} AND tanggal_awal = {tanggal_awal} AND tanggal_akhir = {tanggal_akhir}"
        result_model = db.run_query(query=query_model)
        model_path = result_model[0]['model']
        db.close_connection()

        # Load model dari direktori
        model = joblib.load(model_path)

        # Mengambil data untuk prediksi
        query_data = f"SELECT predicted FROM hasil_prediksi WHERE komoditas_id = {komoditas_id} AND pasar_id = {pasar_id} AND tanggal_awal = {tanggal_awal} AND tanggal_akhir = {tanggal_akhir}"
        result_data = db.run_query(query=query_data)
        db.close_connection()

        data = pd.DataFrame(result_data, columns=['predicted'])
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data.set_index('tanggal', inplace=True)

        # Menambahkan data baru untuk prediksi
        tanggal_akhir = tanggal_akhir + new_predicted_data

        # Persentasse data train dan data test
        PERCENTAGE = 0.8 # Persentase data train adalah 0.8 (80%) untuk saat ini
        train_size = int(len(data_scaled) * PERCENTAGE)
    
        # Normalisasi data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['harga_current']])
        
        # Mempersiapkan data train baru
        new_data_train = data_scaled[:train_size]
        
        # Persiapkan data untuk prediksi
        SEQUENCE_DATA = 60
        new_xTest = []
        new_yTest = []
        for i in range(SEQUENCE_DATA, len(new_data_train)):
            new_xTest.append(new_data_train[i - SEQUENCE_DATA:i, 0])
            # new_yTrain.append(new_data_train[i, 0])
        
        # Convert trained x dan y sebagai numpy array
        new_xTest = np.array(new_xTest)
        # new_yTrain = np.array(new_yTrain)

        # Bentuk ulang x trained data menjadi array 3 dimension
        new_xTest_3d = np.reshape(new_xTest, (new_xTest.shape[0], SEQUENCE_DATA, 1))
            
        # Lakukan prediksi baru
        new_predictions = model.predict(new_xTest_3d)
        new_predictions = scaler.inverse_transform(new_predictions)

        # Evaluasi model menggunakan RMSE
        mse = mean_squared_error(new_yTest, new_predictions)
        rmse = np.sqrt(np.mean(new_predictions - new_yTest) ** 2)
        akurasi = f"Root Mean Squared Error (RMSE) pada data test: {str(rmse)}"

        # Persiapkan hasil prediksi untuk response
        new_predicted_data = pd.DataFrame({'Predicted': new_predictions.flatten()}, index=data.index[-len(new_predictions):])

        return str(model)
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

@app.route(f"{route_prefix}/predictdata", methods=['GET'])
def predictdata():
    try:
        # Mengambil parameter dari request
        # komoditas_id = request.args.get('komoditas_id')
        # pasar_id = request.args.get('pasar_id')
        # tanggal_awal = request.args.get('start_date')
        # tanggal_akhir = request.args.get('end_date')
        # new_predicted_data = int(request.args.get('new_predicted_data'))  # Jumlah hari ke depan untuk prediksi

        komoditas_id = 9
        pasar_id = 101
        tanggal_awal = "2019-01-01"
        tanggal_akhir = "2020-12-31"
        new_data = 61

        # Mengambil model dari database berdasarkan parameter
        query_model = f"SELECT model FROM hasil_prediksi WHERE komoditas_id = '{komoditas_id}' AND pasar_id = '{pasar_id}' AND tanggal_awal = '{tanggal_awal}' AND tanggal_akhir = '{tanggal_akhir}'"
        result_model = db.run_query(query=query_model)
        model_path = result_model[0]['model']
        db.close_connection()
        result_model_df = pd.DataFrame(result_model, columns=['model'])
        
        # Load model dari direktori
        # model = joblib.load(model_path)

        # Mengambil data untuk prediksi
        query_data = f"SELECT predicted_data FROM hasil_prediksi WHERE komoditas_id = '{komoditas_id}' AND pasar_id = '{pasar_id}' AND tanggal_awal = '{tanggal_awal}' AND tanggal_akhir = '{tanggal_akhir}'"
        result_data = db.run_query(query=query_data)
        db.close_connection()

        # Memastikan ada hasil data
        if not result_data:
            return None

        # Mengonversi string JSON ke objek Python
        json_data = result_data[0]['predicted_data']
        data = json.loads(json_data)

        # Ambil data Predicted
        predicted_data = data['predicted_data']['0']
        predicted_data = json.loads(predicted_data)

        # Ambil tanggal dan harga dari data Predicted
        tanggal = list(predicted_data['Predicted'].keys())
        harga = list(predicted_data['Predicted'].values())

        for t, h in zip(tanggal, harga):
            print(f'Tanggal: {t}, Harga: {h}')
        return tanggal, harga
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Fungsi prediksi menggunakan LSTM
@app.route(f"{route_prefix}/predictdata", methods=['GET'])
def predictdata():
    try:
        # Mengambil parameter dari request
        # komoditas_id = request.args.get('komoditas_id')
        # pasar_id = request.args.get('pasar_id')
        # tanggal_awal = request.args.get('start_date')
        # tanggal_akhir = request.args.get('end_date')
        # new_predicted_data = int(request.args.get('new_predicted_data'))  # Jumlah hari ke depan untuk prediksi

        komoditas_id = 9
        pasar_id = 101
        tanggal_awal = "2019-01-01"
        tanggal_akhir = "2020-12-31"
        new_data = 61

        
        # Mengambil model dari database berdasarkan parameter
        # query_model = f"SELECT model FROM hasil_prediksi WHERE komoditas_id = '{komoditas_id}' AND pasar_id = '{pasar_id}' AND tanggal_awal = '{tanggal_awal}' AND tanggal_akhir = '{tanggal_akhir}'"
        # result_model = db.run_query(query=query_model)
        # model_path = result_model[0]['model']
        # db.close_connection()
        # result_model_df = pd.DataFrame(result_model, columns=['model'])
        
        # Load model dari direktori
        # model_path = os.path.join(result_model)
        # model = joblib.load(model_path)

        # Mengambil data untuk prediksi3
        query_data = f"SELECT predicted_data FROM hasil_prediksi WHERE komoditas_id = '{komoditas_id}' AND pasar_id = '{pasar_id}' AND tanggal_awal = '{tanggal_awal}' AND tanggal_akhir = '{tanggal_akhir}'"
        result_data = db.run_query(query=query_data)
        db.close_connection()

        
        # Dataframe data prediksi
        # data = pd.DataFrame(result_data, columns=['predicted_data'])
        # last_data = json.loads(data['predicted_data'].iloc[0])
        # last_predicted_data = last_data['Predicted'] # Data JSON
        # last_actual_data = last_data['Actual'] # Data JSON

        # Mengambil data terakhir untuk prediksi pada masa mendatang
        # last_data = json.loads(data['predicted_data'].iloc[-1])
        # last_actual_data = last_data['Actual']
        # last_predicted_data = last_data['Predicted']

        # # Proses untuk mendapatkan input yang sesuai dengan model LSTM
        # # Normalisasi data
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # last_predicted_data_scaled = scaler.fit_transform(last_predicted_data)

        # # Persentasse data train dan data test
        # PERCENTAGE = 0.8
        # train_size = int(len(last_predicted_data_scaled) * PERCENTAGE)
        
        # # Mempersiapkan data train baru
        # new_data_train = last_predicted_data_scaled[:train_size]
        
        # # Persiapkan data untuk prediksi
        # SEQUENCE_DATA = 60
        # new_xTest = []
        # new_yTest = []
        # for i in range(SEQUENCE_DATA, len(new_data_train)):
        #     new_xTest.append(new_data_train[i - SEQUENCE_DATA:i, 0])
        #     new_yTest.append(new_data_train[i, 0])
        
        # # Convert trained x dan y sebagai numpy array
        # new_xTest = np.array(new_xTest)
        # new_yTest = np.array(new_yTest)

        # # Bentuk ulang x trained data menjadi array 3 dimension
        # new_xTest_3d = np.reshape(new_xTest, (new_xTest.shape[0], SEQUENCE_DATA, 1))

        # # Lakukan prediksi untuk sejumlah hari ke depan
        # for i in range(new_data):
        #     # Lakukan prediksi menggunakan model LSTM
        #     predicted_value = model.predict(new_xTest_3d)
        #     predicted_value = np.reshape(predicted_value, (predicted_value.shape[0], 1))
        #     predicted_value = scaler.inverse_transform(predicted_value)

        #     # Simpan hasil prediksi ke dalam objek JSON respons
        #     last_date = pd.to_datetime(list(last_actual_data.keys())[-1])
        #     next_date = last_date + pd.DateOffset(days=1)
        #     next_date_timestamp = int(next_date.timestamp()) * 1000
        #     last_actual_data[next_date_timestamp] = None  # Tambahkan None karena ini adalah data prediksi
        #     last_predicted_data[next_date_timestamp] = predicted_value[0, 0]

        # # Persiapkan respons JSON
        # response_data = {
        #     "predicted_data": {
        #         "Actual": last_actual_data,
        #         "Predicted": last_predicted_data
        #     }
        # }

        return result_model_df.to_json()
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Load model dari direktori dan prediksi data baru (cara 1)
# if result_model:
#     model_path = result_model[0][0]  # Mengambil path model dari hasil query
#     model = joblib.load(model_path)
#     new_predictions = model.predict(new_xTest_3d)

#     new_data['harga_current'] = new_predictions

#     new_predictions_data = scaler.inverse_transform(new_predictions)
#     new_yTest_original = scaler.inverse_transform(new_yTest.reshape(-1, 1))

#     new_predicted_data = pd.DataFrame({'Predicted': new_predictions_data.flatten(), 'Actual': new_yTest_original.flatten()}, index=data.index[-len(new_predictions):])
#     new_predicted_data_json = new_predicted_data.to_json()

#     return jsonify(new_predicted_data_json)
# else:
#     abort(HTTPStatus.NOT_FOUND, description="Model tidak ditemukan untuk parameter yang diberikan")
        

#         @app.route(f"{route_prefix}/predictdata", methods=['GET'])
# def predictdata():
#     try:
#         # Mengambil parameter dari request
#         # komoditas_id = request.args.get('komoditas_id')
#         # pasar_id = request.args.get('pasar_id')
#         # tanggal_awal = request.args.get('start_date')
#         # tanggal_akhir = request.args.get('end_date')
#         # new_tanggal_akhir = int(request.args.get('new_predicted_data'))  # Jumlah hari ke depan untuk prediksi

#         komoditas_id = 9
#         pasar_id = 12
#         # tanggal_awal = '2019-01-01'
#         # tanggal_akhir = '2020-12-31'
#         tanggal_awal = datetime.strptime('2019-01-01', '%Y-%m-%d').date()
#         tanggal_akhir = datetime.strptime('2020-12-31', '%Y-%m-%d').date()
#         new_tanggal_akhir = 120

#         # Mengambil data untuk prediksi harga pada masa mendatang
#         query_data = f"SELECT tanggal, harga_current FROM daftar_harga WHERE tanggal >= '{tanggal_awal}' AND tanggal <= '{tanggal_akhir}' AND komoditas_id = '{komoditas_id}' AND pasar_id = '{pasar_id}'"
#         result_data = db.run_query(query=query_data)
#         db.close_connection()

#         # Dataframe data prediksi
#         data = pd.DataFrame(result_data, columns=['tanggal', 'harga_current'])
#         # data['tanggal'] = pd.to_datetime(data['tanggal'])
#         # data.set_index('tanggal', inplace=True)

#         # Pengisian missing value dengan rata-rata harga
#         new_avg_harga_current = data['harga_current'].mean()
#         data['harga_current'] = data['harga_current'].replace(0, new_avg_harga_current)
#         data_json = data.to_json(orient='records')

#         # Penambahan data baru ke dalam dataframe (dataframe baru)
#         new_data_tanggal_akhir = tanggal_akhir + timedelta(days=new_tanggal_akhir)
#         new_data = pd.DataFrame({'tanggal': pd.date_range(start=tanggal_akhir + timedelta(days=1), end=new_data_tanggal_akhir), 'harga_current':0})

#         # dataframe baru setelah penambahan data
#         new_dataframe = pd.concat([data, new_data], ignore_index=True)
#         # jumlah_data = f"Jumlah data = {new_dataframe.shape}"

#         # Proses untuk mendapatkan input yang sesuai dengan model LSTM
#         # Normalisasi data
#         data_harga = new_dataframe.iloc[:, new_dataframe.columns.get_loc('harga_current')]
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         new_data_scaled = scaler.fit_transform(data_harga.values.reshape(-1, 1))
#         new_data_json = json.dumps(new_data_scaled.tolist())
#         # jumlah_data = f"Jumlah data = {new_data_scaled.shape}"

#         # Persentase data train dan prepare dataset
#         # PERCENTAGE = 0.8
#         SEQUENCE_DATA = 60
#         # train_size = int(len(new_data_scaled) * PERCENTAGE)
#         # new_data_train = new_data_scaled[:train_size]
#         # new_data_test = new_data_scaled[train_size - SEQUENCE_DATA:]
#         new_data_test = new_data_scaled[-new_tanggal_akhir:]
#         jumlah_data = f"Jumlah data = {new_data_test.shape}"

#         # Pembentukan sequences data / data time series dari data test untuk model prediksi
#         # new_xTest = []
#         # new_yTest = new_data_scaled[train_size:, :]
#         # for i in range(SEQUENCE_DATA, len(new_data_test)):
#         #     new_xTest.append(new_data_test[i - SEQUENCE_DATA:i, 0])

#         new_xTest = []
#         for i in range(len(new_data_test) - SEQUENCE_DATA):
#             new_xTest.append(new_data_test[i:i + SEQUENCE_DATA, 0])

#         # Convert tested x dan y sebagai numpy array 
#         new_xTest= np.array(new_xTest)

#         # Bentuk ulang x tested data menjadi array 3 dimension
#         new_xTest_3d = np.reshape(new_xTest, (new_xTest.shape[0], SEQUENCE_DATA, 1))
#         data_xTest_3d = f"Bentuk data array 3 dimensi xTrain = {new_xTest_3d.shape}"
        
#         # Model prediksi
#         query_model = f"SELECT model FROM hasil_prediksi WHERE komoditas_id = '{komoditas_id}' AND pasar_id = '{pasar_id}' AND tanggal_awal = '{tanggal_awal}' AND tanggal_akhir = '{tanggal_akhir}'"
#         result_model = db.run_query(query=query_model)
#         db.close_connection()
#         # result_model_df = pd.DataFrame(result_model, columns=['model'])

#         # Load model dari direktori dan prediksi data baru (cara 2)
#         model_path = result_model[0][0]
#         model = joblib.load(model_path)
#         new_predictions = model.predict(new_xTest_3d)
            
#         new_data['harga_current'] = new_predictions
            
#         new_predictions_data = scaler.inverse_transform(new_predictions)
#         # new_yTest_original = scaler.inverse_transform(new_yTest.reshape(-1, 1))
            
#         # new_predicted_data = pd.DataFrame({'Predicted': new_predictions.flatten(), 'Actual': new_yTest_original.flatten()}, index=new_data.index[-len(new_predictions):])
#         new_predicted_data = pd.DataFrame({'tanggal': new_data['tanggal'], 'Predicted': new_predictions_data.flatten()})
#         new_predicted_data_json = new_predicted_data.to_json()
            
#         # result_json = new_dataframe.to_json(orient='records')
#         return jsonify(new_predicted_data_json)
#     except pymysql.MySQLError as err:
#         abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
#     except Exception as e:
#         abort(HTTPStatus.BAD_REQUEST, description=str(e))