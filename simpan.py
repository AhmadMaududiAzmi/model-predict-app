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

# Predict function
def run_query(self, query, params=None):
        """Execute SQL query."""
        try:
            if not query or not isinstance(query, str):
                raise Exception()

            if not self.__conn:
                self.__open_connection()
                
            with self.__conn.cursor() as cursor:
                cursor.execute(query, params)
                if 'SELECT' in query.upper():
                    result = cursor.fetchall()
                else:
                    self.__conn.commit()
                    result = f"{cursor.rowcount} row(s) affected."
                cursor.close()

                return result
        except pymysql.MySQLError as sqle:
            raise pymysql.MySQLError(f'Failed to execute query due to: {sqle}')
        except Exception as e:
            raise Exception(f'An exception occured due to: {e}')

@app.route(f"{route_prefix}/predict", methods=['POST'])
def predict():
    try:
        # Get data depending on user input from laravel
        data = request.json
        nm_komoditas = data.get('nm_komoditas')
        nm_pasar = data.get('nm_pasar')
        tanggal_awal = data.get('tanggal_awal')
        tanggal_akhir = data.get('tanggal_akhir')

        # Get data from database for filtering data
        query = "SELECT tanggal, harga_current FROM pertanian.daftar_harga WHERE nm_komoditas = %s AND nm_pasar = %s AND tanggal BETWEEN %s AND %s"
        params = (nm_komoditas, nm_pasar, tanggal_awal, tanggal_akhir)
        records = db.run_query(query=query, params=params)
        db.close_connection()
        
        # Parsing menjadi time series
        dateParse = lambda x: pd.to_datetime(x)
        dataset = pd.DataFrame(records, columns=['tanggal', 'harga_current'])
        dataset['tanggal'] = dataset['tanggal'].apply(dateParse)
        dataset = dataset.sort_values('tanggal')
        dataset.set_index('tanggal', inplace=True)

        # Preprocessing data
        # Perhitungan rata-rata untuk mengisi data harga_current = 0
        avg_harga_current = dataset['harga_current'].mean()
        dataset['harga_current'] = dataset['harga_current'].replace(0, avg_harga_current)
        # Normalisasi data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(dataset[['harga_current']])

        # Dataframe data yang sudah diolah
        df = pd.DataFrame(data_scaled, columns=['harga_current'], index=data.index).reset_index()
        
        
        # ---------------------------------------------------------- #
        # Pembagian dataset menjadi data train dan data test
        # Menggunakan scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Mengubah datasest menjadi array 3 dimension
        SEQUENCE_DATA = 30
        X_train_3d = np.array([X_train[i:i+SEQUENCE_DATA] for i in range(len(X_train) - SEQUENCE_DATA + 1)])
        X_test_3d = np.array([X_test[i:i+SEQUENCE_DATA] for i in range(len(X_test) - SEQUENCE_DATA + 1)])

        # ------- #
        # Menggunakan pytorch
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            # Mengubah datasest menjadi array 3 dimension
        sequence_length = 10  # Panjang sequence yang diinginkan
        X_train_3d = torch.cat([X[train_indices[i]:train_indices[i]+sequence_length].unsqueeze(0) for i in range(len(train_indices) - sequence_length + 1)], dim=0)
        X_test_3d = torch.cat([X[test_indices[i]:test_indices[i]+sequence_length].unsqueeze(0) for i in range(len(test_indices) - sequence_length + 1)], dim=0)

        # ------- #
        # Menggunakan NumPy
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        split = int(0.8 * len(dataset))
        train_indices, test_indices = indices[:split], indices[split:]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
            # Mengubah datasest menjadi array 3 dimension
        SEQUENCE_DATA = 30
        X_train_3d = np.array([X[train_indices[i]:train_indices[i]+SEQUENCE_DATA] for i in range(len(train_indices) - SEQUENCE_DATA + 1)])
        X_test_3d = np.array([X[test_indices[i]:test_indices[i]+SEQUENCE_DATA] for i in range(len(test_indices) - SEQUENCE_DATA + 1)])
        # ---------------------------------------------------------- #

        # ---------------------------------------------------------- #
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
        # ---------------------------------------------------------- #

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

        # ---------------------------------------------------------- #
        # Prepare data test
        data_test = df[train_size:]

        # Pembentukan sequences data / data time series dari data test untuk model prediksi
        xTest = []
        yTest = []
        for i in range(SEQUENCE_DATA, len(data_test)):
            xTest.append(data_test.iloc[i - SEQUENCE_DATA:i]['harga_current'].values)
            yTest.append(data_test.iloc[i]['harga_current'])
        
        # xTest = []
        # yTest = df[train_size:, :]
        # for i in range(SEQUENCE_DATA, len(data_test)):
        #     xTest.append(data_test[i - SEQUENCE_DATA:i]['harga_current'].values)
        
        # Convert tested x dan y sebagai numpy array
        xTest, yTest = np.array(xTest), np.array(yTest)

        # Bentuk ulang x tested data menjadi array 3 dimension
        xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
        # ---------------------------------------------------------- #

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

@app.route(f"{route_prefix}/addPredict", methods=['GET'])
# Penggunaan parameter untuk menambahkan variabel data baru yang berupa hasil prediksi masa mendatang
def addPredict(next_predict, data):
    try:
        # Pengambilan data
        data = request.json
        nm_komoditas = data.get('nm_komoditas')
        nm_pasar = data.get('nm_pasar')
        tanggal_awal = data.get('tanggal_awal')
        tanggal_akhir = data.get('tanggal_akhir')

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

# Loss function
loss = model.evaluate(X_test, y_test)

# Query untuk mencari data yang duplikat
SELECT tanggal, nm_pasar, nm_komoditas, id_komoditas, harga_current, COUNT(*) AS JumlahDuplikat
FROM nama_tabel
GROUP BY tanggal, nm_pasar, nm_komoditas, id_komoditas, harga_current
HAVING COUNT(*) > 1;