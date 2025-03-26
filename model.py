# SOL on XGB Done
import os
import pickle
import pandas as pd
import numpy as np
import requests
import time  # Added for retry logic
from zipfile import ZipFile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import xgboost as xgb
from updater import download_binance_daily_data, download_binance_current_day_data
from config import data_base_path, model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

binance_data_path = os.path.join(data_base_path, "binance")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")

def download_data(token, training_days, region, data_provider):
    if data_provider == "binance":
        files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
        print(f"Downloaded {len(files)} new files for {token}USDT")
        return files
    else:
        raise ValueError("Unsupported data provider")

def format_data(files_btc, files_sol, data_provider):
    if not files_btc or not files_sol:
        print("No files provided for BTCUSDT or SOLUSDT, exiting format_data")
        return
    
    if data_provider == "binance":
        files_btc = sorted([f for f in files_btc if "BTCUSDT" in os.path.basename(f) and f.endswith(".zip")])
        files_sol = sorted([f for f in files_sol if "SOLUSDT" in os.path.basename(f) and f.endswith(".zip")])

    price_df_btc = pd.DataFrame()
    price_df_sol = pd.DataFrame()
    skipped_files = []

    for file in files_btc:
        zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
        if not os.path.exists(zip_file_path):
            continue
        try:
            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                df = pd.read_csv(f, header=None).iloc[:, :11]
                df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"]
                df["date"] = pd.to_datetime(df["end_time"], unit="ms", errors='coerce')
                df = df.dropna(subset=["date"])
                df.set_index("date", inplace=True)
                price_df_btc = pd.concat([price_df_btc, df])
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            skipped_files.append(file)

    for file in files_sol:
        zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
        if not os.path.exists(zip_file_path):
            continue
        try:
            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                df = pd.read_csv(f, header=None).iloc[:, :11]
                df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"]
                df["date"] = pd.to_datetime(df["end_time"], unit="ms", errors='coerce')
                df = df.dropna(subset=["date"])
                df.set_index("date", inplace=True)
                price_df_sol = pd.concat([price_df_sol, df])
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            skipped_files.append(file)

    if price_df_btc.empty or price_df_sol.empty:
        print("No data processed for BTCUSDT or SOLUSDT")
        return

    price_df_btc = price_df_btc.sort_index().loc[~price_df_btc.index.duplicated(keep='last')]
    price_df_sol = price_df_sol.sort_index().loc[~price_df_sol.index.duplicated(keep='last')]
    print(f"BTC rows before concat: {len(price_df_btc)}, SOL rows before concat: {len(price_df_sol)}")
    
    price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
    price_df_sol = price_df_sol.rename(columns=lambda x: f"{x}_SOLUSDT")
    price_df = pd.concat([price_df_btc, price_df_sol], axis=1)
    print(f"Rows after concat: {len(price_df)}")
    print(f"Sample index after concat: {price_df.index[:5].tolist()}")

    if TIMEFRAME != "1m":
        price_df = price_df.resample('5min').agg({
            f"{metric}_{pair}": "last" 
            for pair in ["SOLUSDT", "BTCUSDT"] 
            for metric in ["open", "high", "low", "close"]
        })
        print(f"Rows after resampling to {TIMEFRAME}: {len(price_df)}")
        print(f"Sample index after resampling: {price_df.index[:5].tolist()}")

    price_df = price_df.ffill().bfill()
    print(f"Rows after filling NaNs: {len(price_df)}")

    for pair in ["SOLUSDT", "BTCUSDT"]:
        price_df[f"log_return_{pair}"] = np.log(price_df[f"close_{pair}"].shift(-1) / price_df[f"close_{pair}"])
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 11):
                price_df[f"{metric}_{pair}_lag{lag}"] = price_df[f"{metric}_{pair}"].shift(lag)

    price_df["hour_of_day"] = price_df.index.hour
    price_df["target_SOLUSDT"] = price_df["log_return_SOLUSDT"]
    print(f"Rows after feature engineering: {len(price_df)}")

    price_df = price_df.dropna()
    print(f"Rows after dropna: {len(price_df)}")
    
    if len(price_df) == 0:
        print("No data remains after preprocessing. Check data alignment or gaps.")
        return

    price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Data saved to {training_price_data_path}")

def load_frame(file_path, timeframe):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file {file_path} does not exist.")
    
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    if df.empty:
        raise ValueError(f"Training data file {file_path} is empty.")
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["SOLUSDT", "BTCUSDT"]
        for metric in ["open", "high", "low", "close"]
        for lag in range(1, 11)
    ] + ["hour_of_day"]
    
    X = df[features]
    y = df["target_SOLUSDT"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(len(X) * 0.8)
    if split_idx == 0:
        raise ValueError("Not enough data to split into train and test sets.")
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

def preprocess_live_data(df_btc, df_sol):
    if "date" in df_btc.columns:
        df_btc.set_index("date", inplace=True)
    if "date" in df_sol.columns:
        df_sol.set_index("date", inplace=True)
    
    df_btc = df_btc.rename(columns=lambda x: f"{x}_BTCUSDT" if x != "date" else x)
    df_sol = df_sol.rename(columns=lambda x: f"{x}_SOLUSDT" if x != "date" else x)
    
    df = pd.concat([df_btc, df_sol], axis=1, join='outer')
    print(f"Rows after concat: {len(df)}")

    if TIMEFRAME != "1m":
        df = df.resample('5min').agg({
            f"{metric}_{pair}": "last" 
            for pair in ["SOLUSDT", "BTCUSDT"] 
            for metric in ["open", "high", "low", "close"]
        })
        print(f"Rows after resampling to {TIMEFRAME}: {len(df)}")

    df = df.ffill().bfill()
    print(f"Rows after filling NaNs: {len(df)}")

    for pair in ["SOLUSDT", "BTCUSDT"]:
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 11):
                df[f"{metric}_{pair}_lag{lag}"] = df[f"{metric}_{pair}"].shift(lag)

    df["hour_of_day"] = df.index.hour
    print(f"Rows after feature engineering: {len(df)}")

    df = df.dropna()
    print(f"Rows after dropna: {len(df)}")

    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["SOLUSDT", "BTCUSDT"]
        for metric in ["open", "high", "low", "close"]
        for lag in range(1, 11)
    ] + ["hour_of_day"]
    
    X = df[features]
    if len(X) == 0:
        raise ValueError("No valid data after preprocessing live data.")
    
    with open(scaler_file_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    
    return X_scaled

def train_model(timeframe, file_path=training_price_data_path):
    X_train, X_test, y_train, y_test, scaler = load_frame(file_path, timeframe)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'learning_rate': [0.01, 0.02, 0.05],
        'max_depth': [2, 3],
        'n_estimators': [50, 75, 100],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.5, 0.7],
        'alpha': [10, 20],
        'lambda': [1, 10]
    }
    model = xgb.XGBRegressor(objective="reg:squarederror")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    print(f"Training MAE (log returns): {train_mae:.6f}")
    print(f"Training RMSE (log returns): {train_rmse:.6f}")
    print(f"Training R²: {train_r2:.6f}")

    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    r2 = r2_score(y_test, test_pred)
    print(f"Test MAE (log returns): {mae:.6f}")
    print(f"Test RMSE (log returns): {rmse:.6f}")
    print(f"Test R²: {r2:.6f}")
    
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_file_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Trained model saved to {model_file_path}")
    print(f"Scaler saved to {scaler_file_path}")
    
    return model, scaler

def get_inference(token, timeframe, region, data_provider):
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    # Retry logic for Binance API calls to prevent 500 errors
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            df_btc = download_binance_current_day_data("BTCUSDT", region)
            df_sol = download_binance_current_day_data("SOLUSDT", region)
            ticker_url = f'https://api.binance.{region}/api/v3/ticker/price?symbol=SOLUSDT'
            response = requests.get(ticker_url)
            response.raise_for_status()
            latest_price = float(response.json()['price'])
            break
        except requests.RequestException as e:
            if attempt == max_attempts - 1:
                raise Exception(f"Failed to fetch Binance data after {max_attempts} attempts: {str(e)}")
            print(f"Retry {attempt + 1}/{max_attempts} after error: {e}")
            time.sleep(5)
    
    X_new = preprocess_live_data(df_btc, df_sol)
    log_return_pred = loaded_model.predict(X_new[-1].reshape(1, -1))[0]
    
    predicted_price = latest_price * np.exp(log_return_pred)
    
    print(f"Predicted 5m SOL/USD Log Return: {log_return_pred:.6f}")
    print(f"Latest SOL Price: {latest_price:.2f}")
    print(f"Predicted SOL Price in 5m: {predicted_price:.2f}")
    return predicted_price  # Modified to return absolute price for topic 37
