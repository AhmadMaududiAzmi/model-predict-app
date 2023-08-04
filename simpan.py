# Prediksi menggunakan model LSTM
@app.route(f"{route_prefix}/prediksi", methods=['GET'])
def prediksi():
    try:
        query = "SELECT * FROM pertanian.harga_komoditas WHERE nm_komoditas = 'Gula Pasir Dalam Negri' AND nm_pasar = 'Pasar Senenan'"
        records = db.run_query(query=query)
        db.close_connection()

        # Get data dan parsing menjadi time series
        data = pd.DataFrame(records, columns=['id', 'tanggal', 'nm_pasar', 'nm_komoditas', 'id_komuditas', 'harga_current'])
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data = data.sort_values('tanggal')
        data['tanggal_epoch'] = data['tanggal'].apply(lambda x: x.timestamp())

        # Menghitung rata-rata tiap bulan untuk mengisi values harga_current = 0
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data['bulan'] = data['tanggal'].dt.month
        monthly_avg = data.groupby('bulan')['harga_current'].mean()
        data.loc[data['harga_current'] == 0, 'harga_current'] = data['bulan'].map(monthly_avg)
        
        # Normalisasi menggunakan MinMaxScaler()
        # scaler = MinMaxScaler()
        # data['harga_scaled'] = scaler.fit_transform(data[['harga_current']])
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # # Split data ke data train dan data test
        X = data['tanggal_epoch'].values.reshape(-1, 1)
        y = data['harga_scaled'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

        # # Model LSTM
        # model = Sequential()
        # model.add(LSTM(64, return_sequences=True, input_shape=(1, 1)))
        # model.add(LSTM(64))
        # model.add(Dense(25))
        # model.add(Dense(1))
        # model.compile(optimizer='adam', loss='mse')

        # # Train data
        # model.fit(X_train, y_train, epochs=150, batch_size=32)

        # Get model dari joblib (model yang disimpan)
        loaded_model = joblib.load('trained_model.joblib')

        # Mengambil tanggal terakhir dari data historis
        last_date = data['tanggal'].iloc[-1]

        # Membuat data tanggal untuk 30 hari ke depan
        future_dates = [last_date + timedelta(days=i) for i in range(1, 60)]

        # Mengubah data tanggal menjadi nilai epoch
        future_dates_epoch = np.array([date.timestamp() for date in future_dates]).reshape(-1, 1)

        # Melakukan prediksi menggunakan model LSTM
        # y_pred_scaled = model.predict(future_dates_epoch)
        y_pred_scaled = loaded_model.predict(future_dates_epoch)

        # Mengembalikan hasil prediksi ke dalam skala semula
        y_pred = scaler.inverse_transform(y_pred_scaled)

        prediction_results = {
            'Tanggal Prediksi': [date.strftime('%Y-%m-%d') for date in future_dates],
            'Harga Prediksi': y_pred.flatten().tolist()
        }

        return jsonify(prediction_results)
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Melakukan train model dengan menggunakan data train
@app.route(f"{route_prefix}/trainData", methods=["GET", "POST"])
def trainData():
    try:
        if request.method == "GET":
            return
        if request.method == 'POST':
             # Get data dari lemparan laravel
            data_train = request.get_json()

            # Convert data train ke DataFrame
            data_train = pd.DataFrame(data_train)

            # Preprocess data train
            X_train = data_train.drop('harga_current', axis=1)
            y_train = data_train['harga_current']

            # Normalize the data train using MinMaxScaler
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            y_train_scaled = scaler.transform(y_train.values.reshape(-1, 1))

            # Reshape the input data for LSTM (assuming input shape (samples, time steps, features))
            X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(units=128, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
            model.add(Dense(units=1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train_reshaped, y_train_scaled, epochs=100, batch_size=32)

            # Save the trained model
            model.save('trained_model.h5')

            return "Model trained and saved successfully." 
    except pymysql.MySQLError as err:
        abort(HTTPStatus.INTERNAL_SERVER_ERROR, description=str(err))
    except Exception as e:
        abort(HTTPStatus.BAD_REQUEST, description=str(e))

# Loss function
loss = model.evaluate(X_test, y_test)

# Query untuk mencari data yang duplikat
SELECT tanggal, nm_pasar, nm_komoditas, id_komoditas, harga_current, COUNT(*) AS JumlahDuplikat
FROM nama_tabel
GROUP BY tanggal, nm_pasar, nm_komoditas, id_komoditas, harga_current
HAVING COUNT(*) > 1;
