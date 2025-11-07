from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input

# ================================
#  Ayarlar
# ================================
CLASSES = ["healthy", "green", "rotten"]
IMG_SIZE = (128, 128)

def pick_model_path():
    ft = Path("models/potato_model_ft.keras")
    base = Path("models/potato_model.keras")
    return str(ft if ft.exists() else base)

# (Opsiyonel) GPU bellek büyütmesi: küçük scriptlerde VRAM'i komple kilitlemez
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ================================
#  Görsel bütünlüğü kontrolü
# ================================
def quick_image_check(root="data/dataset/train", limit=5):
    """
    Dataset'te rastgele birkaç görseli açarak dosya bozulması olup olmadığını kontrol eder.
    """
    root = Path(root)
    cnt = 0
    for p in root.rglob("*.*"):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue
        try:
            with Image.open(p) as im:
                ImageOps.exif_transpose(im).convert("RGB")
            cnt += 1
            if cnt >= limit:
                break
        except Exception as e:
            print(f"Bozuk dosya: {p} -> {e}")
    print(f"Açılan örnek sayısı: {cnt}")

# ================================
#  Tek görsel tahmini
# ================================
def predict_image(path, img_size=IMG_SIZE, topk=3, verbose=True):
    """
    Tek bir görseli yükler, modeli çalıştırır ve tahmin döndürür.
    Return: (pred_label, pred_conf, probs_dict)
    """
    model_path = pick_model_path()
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Model yüklenemedi: {model_path} -> {e}")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Görsel bulunamadı: {path}")

    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize(img_size)
    except Exception as e:
        raise RuntimeError(f"Görsel açılamadı: {path} -> {e}")

    x = np.array(im, dtype=np.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    probs = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

    if verbose:
        print(f"Model: {model_path}")
        print(f"Pred:  {CLASSES[idx]} (conf={conf:.2f})")
        # Top-K
        order = np.argsort(preds)[::-1][:topk]
        for i in order:
            print(f"  - {CLASSES[i]:7s}: {preds[i]:.3f}")

    return CLASSES[idx], conf, probs

# ================================
#  Manuel kullanım
# ================================
if __name__ == "__main__":
    # quick_image_check()
    # predict_image('data/dataset/val/green/example.jpg')
    pass
