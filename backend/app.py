import tensorflow as tf
from .utils.preprocess import preprocess_audio
from .utils.youtube import load_audio_from_youtube
from .utils.audio import load_audio_from_mp3
from pathlib import Path
import joblib
import numpy as np
from flask import Flask, request, jsonify
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__, static_folder="../frontend")

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent

MODEL_PATH = project_root / "resources" / \
    "models" / "mel_2048_cnn_lstm_model_d.h5"
SCALER_PATH = project_root / "resources" / \
    "scalers" / "scaler_aug_mel_2048_d.gz"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
    "dangdut"
]


@app.route("/")
def index():
    return app.send_static_file('index.html')


@app.route("/api/predict-file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    audio_bytes = file.read()

    y, sr = load_audio_from_mp3(audio_bytes)
    X = preprocess_audio(y, sr, scaler)
    preds = model.predict(X)
    avg_preds = np.mean(preds, axis=0)
    dict_avg_preds = dict(zip(LABELS, avg_preds))
    idx = np.argmax(avg_preds)

    return jsonify({
        "genre": LABELS[idx],
        "probabilities": dict_avg_preds
    })


@app.route("/api/predict-youtube", methods=["POST"])
def predict_youtube():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing YouTube url"}), 400

    y, sr = load_audio_from_youtube(data["url"])
    X = preprocess_audio(y, sr, scaler)
    preds = model.predict(X)
    avg_preds = np.mean(preds, axis=0)
    dict_avg_preds = dict(zip(LABELS, avg_preds))
    idx = np.argmax(avg_preds)

    return jsonify({
        "genre": LABELS[idx],
        "probabilities": dict_avg_preds
    })
