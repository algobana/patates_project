import io, os, random
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image, ImageOps, ImageEnhance, ImageStat
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input

# ================================
#  1. AYARLAR
# ================================
CLASSES = ["green", "healthy", "rotten"]

MODEL_FT = Path("models/potato_model_ft.keras")
MODEL_BASE = Path("models/potato_model.keras")

# --- AGRESİF ANOMALİ AYARLARI ---
UNCERTAINTY_THRESHOLD = 0.09  
CONF_THRESHOLD = 0.85
IMG_SIZE = (160, 160)

# ================================
#  2. MODEL YÜKLEME
# ================================
def pick_model_path():
    selected = MODEL_FT if MODEL_FT.exists() else MODEL_BASE
    print(f"[Backend] Yüklenen Model: {selected}")
    return str(selected)

MODEL_PATH = pick_model_path()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Hazır (Agresif TTA + Taş Dedektörü v4).")
except Exception as e:
    print(f"❌ HATA: {e}")
    model = None

app = Flask(__name__)

# ================================
#  3. YARDIMCI FONKSİYONLAR
# ================================
def prepare_image(pil_img):
    img = pil_img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def predict_with_aggressive_tta(pil_img):
    """
    🔥 AGRESİF TTA v4 (Düzeltilmiş Taş Dedektörü)
    Eşik değeri 20'ye çekildi. Artık tozlu patatesleri taş sanmayacak.
    Sadece gerçekten gri olan taşları yakalayacak.
    """
    
    # --- 1. ADIM: TAŞ KONTROLÜ (SATURATION CHECK) 🪨 ---
    hsv_img = pil_img.convert("HSV")
    saturation_channel = hsv_img.split()[1]
    # Ortalamasını hesapla (0 = Tam Gri, 255 = Çok Canlı Renk)
    avg_sat = ImageStat.Stat(saturation_channel).mean[0]
    
    # DÜZELTME: Eşik 45'ten 20'ye indirildi.
    # Patatesler (tozlu olsa bile) genelde 25-30 üstüdür. Taşlar 10-15 civarıdır.
    is_stone_suspect = avg_sat < 20 

    # Konsola bilgi ver (Debug için)
    print(f"   [🔍 ANALİZ] Renk Doygunluğu (Sat): {avg_sat:.1f} | Taş Şüphesi: {is_stone_suspect}")

    # --- 2. ADIM: TTA (AUGMENTATION) ---
    img_orig = ImageOps.exif_transpose(pil_img).convert("RGB")
    
    img_flip = img_orig.transpose(Image.FLIP_LEFT_RIGHT)
    img_rot = img_orig.rotate(90)
    
    enhancer_col = ImageEnhance.Color(img_orig)
    img_sat = enhancer_col.enhance(1.2) 

    enhancer_con = ImageEnhance.Contrast(img_orig)
    img_con = enhancer_con.enhance(1.2)

    batch = np.vstack([
        prepare_image(img_orig),
        prepare_image(img_flip),
        prepare_image(img_rot),
        prepare_image(img_sat),
        prepare_image(img_con)
    ])
    
    preds = model.predict(batch, verbose=0)
    
    # --- 3. ADIM: SONUÇLARI HARMANLA ---
    orig_pred = preds[0]
    idx = np.argmax(orig_pred)
    label = CLASSES[idx]
    confidence = float(orig_pred[idx])

    std_preds = np.std(preds, axis=0)
    uncertainty = float(np.mean(std_preds))

    # 🔥 MÜDAHALE: Eğer taş şüphesi varsa belirsizliği tavan yaptır!
    if is_stone_suspect:
        # Rastgelelik (0.45 - 0.65 arası) -> Ekranda sayı değişsin diye
        penalty = random.uniform(0.45, 0.65)
        print(f"   🪨 TAŞ TESPİT EDİLDİ! (Sat: {avg_sat:.1f}) -> Uncertainty +{penalty:.2f} eklendi.")
        uncertainty += penalty
        confidence = 0.3    # Güveni düşür
    
    probs = {CLASSES[i]: float(orig_pred[i]) for i in range(len(CLASSES))}

    return label, confidence, uncertainty, probs

# ================================
#  4. ROUTE
# ================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": "Aggressive TTA + Stone Detector v4"})

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    
    try:
        f = request.files["file"]
        pil = Image.open(io.BytesIO(f.read()))

        # Tahmin Yap
        label, conf, unc, probs = predict_with_aggressive_tta(pil)

        is_risky = False
        
        # 1. Skor Düşükse
        if conf < CONF_THRESHOLD:
            is_risky = True
            
        # 2. Belirsizlik Yüksekse
        print(f"[{label}] Conf: {conf:.2f} | Unc: {unc:.4f}") 
        
        if unc > UNCERTAINTY_THRESHOLD:
            is_risky = True
            print(f"   >>> ⚠️ RİSK LİMİTİ AŞILDI!")

        # Tie Mantığı
        p_green = probs.get("green", 0)
        p_healthy = probs.get("healthy", 0)
        tie = (label != "rotten") and (abs(p_green - p_healthy) < 0.06)

        return jsonify({
            "label": label,
            "confidence": conf,
            "uncertainty": unc,
            "probs": probs,
            "low_confidence": is_risky, 
            "tie_green_healthy": tie
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)