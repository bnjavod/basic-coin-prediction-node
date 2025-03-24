import json
import os
import time
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

app = Flask(__name__)

def update_data():
    print("Starting data update process...")
    # Clear all data to force refresh
    data_dir = os.path.join(os.getcwd(), "data", "binance")
    price_data_file = os.path.join(os.getcwd(), "data", "price_data.csv")
    model_file = model_file_path
    scaler_file = scaler_file_path
    for path in [data_dir, price_data_file, model_file, scaler_file]:
        if os.path.exists(path):
            if os.path.isdir(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
            else:
                os.remove(path)
            print(f"Cleared {path}")
    
    print("Downloading BTC data...")
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    print("Downloading SOL data...")
    files_sol = download_data("SOL", TRAINING_DAYS, REGION, DATA_PROVIDER)
    if not files_btc or not files_sol:
        print("No data files downloaded. Skipping format_data and training.")
        return
    print("Formatting data...")
    format_data(files_btc, files_sol, DATA_PROVIDER)
    print("Training model...")
    train_model(TIMEFRAME)
    print("Data update and training completed.")

@app.route("/inference/<string:token>")
def generate_inference(token):
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else f"Token {token} not supported, expected {TOKEN}"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')
    try:
        if not os.path.exists(model_file_path):
            raise FileNotFoundError("Model file not found. Please run update first to train the model.")
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(str(inference), status=200, mimetype='text/plain')  # Returns log return as text
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    try:
        update_data()
        return "0"
    except Exception as e:
        print(f"Update failed: {str(e)}")
        return "1"

if __name__ == "__main__":
    update_data()
    while not os.path.exists(model_file_path) or not os.path.exists(scaler_file_path):
        print("Waiting for model and scaler files to be generated...")
        time.sleep(5)
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
