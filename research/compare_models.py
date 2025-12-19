import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input

# AYARLAR
DATA_DIR = "data/dataset"
IMG_SIZE_ML = (64, 64)   # ML modelleri için (Hız için küçük)
IMG_SIZE_CNN = (160, 160) # CNN için (Orijinal boyut)
CLASSES = ["green", "healthy", "rotten"]

def load_data_for_ml(split_name):
    """Klasik ML (Decision Tree, NB) için veriyi düzleştirir."""
    X, y = [], []
    path = os.path.join(DATA_DIR, split_name)
    print(f"Loading {split_name} data for ML models...")
    for label_idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(path, cls)
        if not os.path.exists(cls_path): continue
        for img_name in os.listdir(cls_path):
            try:
                img = cv2.imread(os.path.join(cls_path, img_name))
                if img is None: continue
                img = cv2.resize(img, IMG_SIZE_ML)
                X.append(img.flatten()) # Resmi vektöre çevir
                y.append(label_idx)
            except Exception as e:
                pass
    return np.array(X), np.array(y)

def preprocess_cnn_image(path):
    """CNN için resmi yükler ve -1,1 arasına çeker."""
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE_CNN)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def run_comparison():
    # --- 1. VERİ YÜKLEME (ML Modelleri İçin) ---
    X_train, y_train = load_data_for_ml("train")
    X_test, y_test = load_data_for_ml("test")
    
    # --- 2. VERİ YÜKLEME (CNN İçin - Düzeltilmiş Kısım) ---
    print("Loading test data for CNN...")
    X_test_cnn = []
    y_test_cnn_idx = []
    
    path = os.path.join(DATA_DIR, "test")
    for label_idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(path, cls)
        if not os.path.exists(cls_path): continue
        for img_name in os.listdir(cls_path):
            try:
                full_path = os.path.join(cls_path, img_name)
                processed_img = preprocess_cnn_image(full_path)
                X_test_cnn.append(processed_img)
                y_test_cnn_idx.append(label_idx)
            except: pass
            
    X_test_cnn = np.array(X_test_cnn)
    y_test_cnn_idx = np.array(y_test_cnn_idx)

    # --- 3. MODEL EĞİTİMLERİ ---
    
    # Model A: Decision Tree
    print("\n--- Training Decision Tree ---")
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_probs = dt_model.predict_proba(X_test) # Hata buradaydı, artık hesaplanıyor
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_preds):.4f}")

    # Model B: Naive Bayes
    print("\n--- Training Naive Bayes ---")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    nb_probs = nb_model.predict_proba(X_test)
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_preds):.4f}")

    # Model C: CNN (MobileNetV2)
    print("\n--- Loading CNN Model ---")
    try:
        cnn_model = tf.keras.models.load_model("models/potato_model.keras")
        # Eski ft dosyası varsa onu yüklemesin diye direkt ismini verdik
    except:
        print("HATA: 'models/potato_model.keras' bulunamadı!")
        return

    cnn_probs = cnn_model.predict(X_test_cnn)
    cnn_preds = np.argmax(cnn_probs, axis=1)
    print(f"CNN Accuracy: {accuracy_score(y_test_cnn_idx, cnn_preds):.4f}")

    # --- 4. ROC EĞRİLERİ ---
    plt.figure(figsize=(10, 6))
    
    def plot_roc(y_true, y_probs, name):
        # Multiclass ROC için basit bir yaklaşım (micro-average veya sınıf bazlı değil, genel başarı)
        # Hoca detay istemediyse en temiz yöntem binary gibi çizmektir ama 3 sınıf var.
        # Burada her sınıf için ayrı çizmek yerine 'macro' ortalama alıyoruz.
        try:
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            n_classes = 3
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Tüm sınıfların ortalamasını alarak tek bir çizgi çizelim (daha temiz grafik)
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            
            mean_auc = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, label=f'{name} (Mean AUC = {mean_auc:.2f})', linewidth=2)
        except Exception as e:
            print(f"ROC Hatası ({name}): {e}")

    plot_roc(y_test, dt_probs, "Decision Tree")
    plot_roc(y_test, nb_probs, "Naive Bayes")
    plot_roc(y_test_cnn_idx, cnn_probs, "CNN (MobileNetV2)")

    plt.plot([0, 1], [0, 1], 'k--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Multi-class Mean)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("roc_comparison.png")
    print("\n✅ roc_comparison.png oluşturuldu.")

    # --- 5. RAPOR TABLOSU ---
    print("\n====== REPORT DATA ======")
    print(f"{'Algorithm':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    
    results = [
        ("Decision Tree", y_test, dt_preds),
        ("Naive Bayes", y_test, nb_preds),
        ("CNN (Deep)", y_test_cnn_idx, cnn_preds)
    ]
    
    for name, true, pred in results:
        acc = accuracy_score(true, pred)
        p, r, f, _ = precision_recall_fscore_support(true, pred, average='weighted', zero_division=0)
        print(f"{name:<20} | {acc:.4f}     | {p:.4f}     | {r:.4f}     | {f:.4f}")

if __name__ == "__main__":
    run_comparison()