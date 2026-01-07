import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from pathlib import Path
from keras.applications.mobilenet_v2 import preprocess_input

# ================================
#  AYARLAR
# ================================
CLASSES = ["healthy", "green", "rotten"]
IMG_SIZE = (160, 160) 
UNCERTAINTY_THRESHOLD = 0.15  # Belirsizlik eşiği (Hassasiyete göre oyna)

def pick_model_path():
    """Önce fine-tune (ft) modelini, yoksa base modeli seçer."""
    ft = Path("models/potato_model_ft.keras")
    base = Path("models/potato_model.keras")
    selected = ft if ft.exists() else base
    print(f"[Sistem] Seçilen Model: {selected}")
    return str(selected)

# GPU Bellek Ayarı (Opsiyonel)
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
except:
    pass

# Modeli global olarak yükleyelim (Sürekli diskten okumasın, hızlanır)
MODEL_PATH = pick_model_path()
try:
    GLOBAL_MODEL = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ Model Yükleme Hatası: {e}")
    GLOBAL_MODEL = None

# ================================
#  YENİLİK: Monte Carlo Dropout 🧠
# ================================
def predict_with_uncertainty(model, img_array, n_iter=10):
    """
    Modeli 'training=True' modunda N kere çalıştırır.
    Dropout katmanları aktif olduğu için her seferinde farklı sonuç verir.
    Bu sonuçların standart sapması (std) bize 'BELİRSİZLİĞİ' verir.
    """
    # (N, 160, 160, 3) boyutunda çoğalt
    pixels_repeated = np.repeat(img_array, n_iter, axis=0)
    
    # training=True -> Dropout AKTİF (Bayesyen Yaklaşım)
    preds = model(pixels_repeated, training=True) 
    
    # İstatistikleri hesapla
    prediction_mean = np.mean(preds, axis=0)  # Ortalama tahmin
    uncertainty = np.std(preds, axis=0)       # Standart sapma (Belirsizlik)
    
    # En yüksek sınıfa ait belirsizlik değeri
    top_class_idx = np.argmax(prediction_mean)
    confidence_score = prediction_mean[top_class_idx]
    uncertainty_score = uncertainty[top_class_idx]
    
    return top_class_idx, confidence_score, uncertainty_score, prediction_mean

# ================================
#  Ana Tahmin Fonksiyonu
# ================================
def predict_image(path, verbose=True):
    if GLOBAL_MODEL is None:
        return "Error", 0.0, 0.0

    p = Path(path)
    if not p.exists():
        print(f"Resim bulunamadı: {path}")
        return "Error", 0.0, 0.0

    # Resmi Hazırla
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize(IMG_SIZE)
            img_array = np.array(im, dtype=np.float32)
            img_array = preprocess_input(img_array) # -1, 1 normalizasyon
            img_array = np.expand_dims(img_array, axis=0) # (1, 160, 160, 3)
    except Exception as e:
        print(f"Resim işleme hatası: {e}")
        return "Error", 0.0, 0.0

    # 🔥 YENİLİKÇİ TAHMİN (MC DROPOUT)
    class_idx, conf, unc, all_probs = predict_with_uncertainty(GLOBAL_MODEL, img_array, n_iter=20)
    
    label = CLASSES[class_idx]
    
    # Karar Mekanizması: Eğer belirsizlik çok yüksekse 'UNCERTAIN' de.
    final_decision = label
    if unc > UNCERTAINTY_THRESHOLD:
        final_decision = "UNCERTAIN"  # Yeni Sınıf!
        if verbose: print(f"⚠️ DİKKAT: Model kararsız! (Belirsizlik: {unc:.4f})")

    if verbose:
        print(f"📸 Görüntü: {p.name}")
        print(f"🧠 Tahmin: {label} (Güven: {conf:.4f})")
        print(f"📉 Belirsizlik (Varyans): {unc:.4f}")
        print(f"📢 Nihai Karar: {final_decision}")
        print("-" * 30)

    return final_decision, float(conf), float(unc)

if __name__ == "__main__":
    # Test et
    predict_image("data/dataset/val/rotten/rot1.jpg") # Yolunu kendine göre ayarla