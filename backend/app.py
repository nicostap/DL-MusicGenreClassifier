from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
from pathlib import Path

from utils.audio import load_audio_from_mp3
from utils.youtube import load_audio_from_youtube
from utils.preprocess import preprocess_audio

app = Flask(__name__)

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent

MODEL_PATH = project_root / "resources" / "models" / "mel_2048_cnn_lstm_model_d.h5"
SCALER_PATH = project_root / "resources" / "scalers" / "scaler_aug_mel_2048_d.gz"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
    "dangdut"
]


@app.route("/")
def index():
    # TO:DO Serve HTML
    return {"status": "OK", "message": "Music genre classifier running"}


@app.route("/api/predict-file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    audio_bytes = file.read()

    y, sr = load_audio_from_mp3(audio_bytes)
    x = preprocess_audio(y, sr)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)

    return jsonify({
        "genre": LABELS[idx],
        "probabilities": preds.tolist()
    })


@app.route("/api/predict-youtube", methods=["POST"])
def predict_youtube():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing YouTube url"}), 400

    y, sr = load_audio_from_youtube(data["url"])
    x = preprocess_audio(y, sr, scaler)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)

    return jsonify({
        "genre": LABELS[idx],
        "probabilities": preds.tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
