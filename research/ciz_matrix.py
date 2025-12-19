import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.applications.mobilenet_v2 import preprocess_input

# AYARLAR
DATA_DIR = "data/dataset"
IMG_SIZE = (160, 160)
CLASSES = ["green", "healthy", "rotten"]
MODEL_PATH = "models/potato_model.keras"

def ciz_matrix():
    print("Model yükleniyor...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except:
        print("Model bulunamadı! Lütfen önce train yap.")
        return

    print("Test verileri hazırlanıyor...")
    X_test = []
    y_true = []

    # Test klasörünü oku
    test_path = os.path.join(DATA_DIR, "test")
    for label_idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(test_path, cls)
        if not os.path.exists(cls_path): continue
        
        for img_name in os.listdir(cls_path):
            try:
                full_path = os.path.join(cls_path, img_name)
                # Resmi yükle ve işle
                img = tf.keras.utils.load_img(full_path, target_size=IMG_SIZE)
                img_arr = tf.keras.utils.img_to_array(img)
                img_arr = preprocess_input(img_arr)
                
                X_test.append(img_arr)
                y_true.append(label_idx)
            except: pass

    X_test = np.array(X_test)
    y_true = np.array(y_true)

    print("Tahmin yapılıyor...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Matrisi Çiz ve Kaydet
    print("Grafik çiziliyor...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap='Blues', values_format='d') # 'd' tam sayı olarak gösterir
    
    plt.title("Confusion Matrix (CNN Model)")
    plt.savefig("confusion_matrix.png") # Dosyayı kaydeder
    print(f"\n✅ BAŞARILI! 'confusion_matrix.png' dosyası oluşturuldu.")
    print(f"Toplam Test Resmi: {len(y_true)}")

if __name__ == "__main__":
    ciz_matrix()