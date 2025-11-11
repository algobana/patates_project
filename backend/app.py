# backend/app.py
import io, os
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input

# ------------ Config ------------
CLASSES = ["green", "healthy", "rotten"]
MODEL_FT = Path("models/potato_model_ft.keras")
MODEL_BASE = Path("models/potato_model.keras")

# Eşikler
CONF_THRESHOLD = 0.45      # low_confidence için alt sınır
TIE_DELTA = 0.06           # healthy vs green yakınlığı için eşik
USE_TTA = True             # test-time augmentation (horizontal flip)

# ------------ Model ------------
MODEL_PATH = str(MODEL_FT if MODEL_FT.exists() else MODEL_BASE)
model = tf.keras.models.load_model(MODEL_PATH)

# Model giriş boyutunu otomatik al (mismatch olmaz)
H, W = model.input_shape[1], model.input_shape[2]
IMG_SIZE = (W, H)

app = Flask(__name__)

# ------------ Helpers ------------
def to_batch(pil_img, img_size=IMG_SIZE):
    """PIL -> model girişi (1, H, W, 3), preprocess_input ile [-1,1]."""
    arr = np.array(pil_img.convert("RGB").resize(img_size), dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def predict_with_flags(pil_img):
    """Tek görsel için tahmin + low_confidence + tie_green_healthy bayrakları."""
    p0 = model.predict(to_batch(pil_img), verbose=0)[0]
    if USE_TTA:
        p1 = model.predict(to_batch(pil_img.transpose(Image.FLIP_LEFT_RIGHT)), verbose=0)[0]
        p = (p0 + p1) / 2.0
    else:
        p = p0

    idx = int(np.argmax(p))
    conf = float(np.max(p))
    probs = {
        "healthy": float(p[0]),
        "green":   float(p[1]),
        "rotten":  float(p[2]),
    }

    return {
        "label": CLASSES[idx],
        "confidence": float(conf),
        "probs": probs,
        "low_confidence": bool(conf < CONF_THRESHOLD),
        # Tie sadece green-healthy arasında geçerli, rotten hariç
        "tie_green_healthy": bool(
            (max(p[0], p[1]) > p[2]) and abs(float(p[0]) - float(p[1])) < TIE_DELTA
        ),
    }


# ------------ Routes ------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH,
        "img_size": IMG_SIZE,
        "classes": CLASSES,
        "use_tta": USE_TTA,
        "conf_threshold": CONF_THRESHOLD,
        "tie_delta": TIE_DELTA,
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    try:
        data = request.files["file"].read()
        pil = Image.open(io.BytesIO(data))
        out = predict_with_flags(pil)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    if "files" not in request.files:
        return jsonify({"error": "no files"}), 400
    results = []
    for f in request.files.getlist("files"):
        try:
            pil = Image.open(f.stream)
            out = predict_with_flags(pil)
            out["filename"] = getattr(f, "filename", "")
            results.append(out)
        except Exception as e:
            results.append({"filename": getattr(f, "filename", ""), "error": str(e)})
    return jsonify({"results": results})

if __name__ == "__main__":
    # Geliştirme sunucusu
    app.run(host="0.0.0.0", port=5000, debug=False)
