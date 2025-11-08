import io, os
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input

# ---- Config ----
CLASSES = ["healthy", "green", "rotten"]
MODEL_FT = Path("models/potato_model_ft.keras")
MODEL_BASE = Path("models/potato_model.keras")
IMG_SIZE = (128, 128)

# XLA uyarıları can sıkarsa aç:
# tf.config.optimizer.set_jit(False)

# ---- Model yükleme (lazy değil; server start'ta) ----
MODEL_PATH = str(MODEL_FT if MODEL_FT.exists() else MODEL_BASE)
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

def prepare_image(pil_img, target=IMG_SIZE):
    img = pil_img.convert("RGB").resize(target)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH, "classes": CLASSES})

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    try:
        file = request.files["file"].read()
        pil = Image.open(io.BytesIO(file))
        x = prepare_image(pil)
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        return jsonify({
            "label": CLASSES[idx],
            "confidence": conf,
            "probs": {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    # Çoklu dosya desteği: 'files' alanında birden fazla görsel
    if "files" not in request.files:
        return jsonify({"error": "no files"}), 400
    results = []
    for f in request.files.getlist("files"):
        try:
            pil = Image.open(f.stream)
            x = prepare_image(pil)
            preds = model.predict(x, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            results.append({
                "filename": getattr(f, "filename", ""),
                "label": CLASSES[idx],
                "confidence": conf,
                "probs": {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
            })
        except Exception as e:
            results.append({"filename": getattr(f, "filename", ""), "error": str(e)})
    return jsonify({"results": results})

if __name__ == "__main__":
    # Geliştirme modu
    app.run(host="0.0.0.0", port=5000, debug=False)
