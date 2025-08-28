import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import joblib
from utils.data_preparation import prepare_data

# Подготовка данных
df_d2 = prepare_data()

# Подготовка признаков и целевой переменной
features = ['gold', 'silver', 'palladium', 'key interest rate', 'inflation', 'curs eur', 'brent (in usd)', 
            'curs usd_lag1', 'curs usd_lag2', 'curs usd_lag3', 'curs usd_lag4', 'curs usd_lag5', 
            'curs usd_lag6', 'curs usd_lag7']
X = df_d2[features]
y = df_d2['curs usd']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение моделей
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)

# Нейронная сеть
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# Сохранение моделей
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(svr_model, 'svr_model.joblib')
nn_model.save('nn_model.h5')
joblib.dump(scaler, 'scaler.joblib')

# Функция предсказания
def predict_combined(data):
    data_scaled = scaler.transform(data)
    lr_pred = lr_model.predict(data_scaled)
    rf_pred = rf_model.predict(data_scaled)
    svr_pred = svr_model.predict(data_scaled)
    nn_pred = nn_model.predict(data_scaled, verbose=0).flatten()
    combined_pred = (lr_pred + rf_pred + svr_pred + nn_pred) / 4
    return combined_pred

# Функция для расчета метрик
def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
    from math import sqrt
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return {'R²': r2, 'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse}

# Пример расчета метрик на тестовых данных
y_pred = predict_combined(X_test)
metrics = calculate_metrics(y_test, y_pred)