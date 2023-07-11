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