import os
import tensorflow as tf
from pathlib import Path
from keras import layers
from keras.applications.mobilenet_v2 import preprocess_input

DATA_DIR = "data/dataset"
BEST_IN  = "models/potato_model.keras"        # önceki en iyi model
BEST_OUT = "models/potato_model_ft.keras"     # fine-tune sonrası
CLASSES  = ["healthy","green","rotten"]
BATCH_SIZE = 24        # 160px için güvenli
EPOCHS = 10
UNFREEZE_TAIL = 80     # <<< son ~80 katmanı aç

def build_ds(img_size):
    def ds(split, shuffle):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(DATA_DIR, split),
            labels="inferred", label_mode="categorical",
            image_size=img_size, batch_size=BATCH_SIZE, shuffle=shuffle
        )
        AUTOTUNE = tf.data.AUTOTUNE
        def prep(x,y): return preprocess_input(tf.cast(x, tf.float32)), y
        return ds.map(prep, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    return ds("train", True), ds("val", False)

def compute_class_weights(train_dir=os.path.join(DATA_DIR,"train")):
    counts = {}
    for c in CLASSES:
        p = os.path.join(train_dir, c)
        n = len([f for f in os.listdir(p) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
        counts[c] = max(1, n)
    total = sum(counts.values())
    weights = {i: total/(len(CLASSES)*counts[cls]) for i, cls in enumerate(CLASSES)}
    print("[class_counts]", counts)
    print("[class_weights]", weights)
    return weights

def unfreeze_tail(mobilenet_model, n_tail=UNFREEZE_TAIL):
    n = len(mobilenet_model.layers)
    cut = max(0, n - n_tail)
    for i, l in enumerate(mobilenet_model.layers):
        # BatchNorm'ları genelde sabit bırakmak daha stabildir
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
        else:
            l.trainable = (i >= cut)

if __name__ == "__main__":
    # 1) Modeli yükle ve IMG_SIZE’ı modelden al (mismatch riskini sıfırlar)
    model = tf.keras.models.load_model(BEST_IN)
    h, w = model.input_shape[1], model.input_shape[2]
    IMG_SIZE = (h, w)
    print(f"[info] Model input size: {IMG_SIZE}")

    # 2) Datasetleri aynı boyutta kur
    train_ds, val_ds = build_ds(IMG_SIZE)

    # 3) MobileNetV2 gövdesini bul
    base = None
    for l in model.layers:
        if isinstance(l, tf.keras.Model) and "mobilenetv2" in l.name.lower():
            base = l; break
    if base is None:
        raise RuntimeError("MobileNetV2 tabanı model içinde bulunamadı.")

    # 4) Son ~80 katmanı aç (BN sabit)
    unfreeze_tail(base, n_tail=UNFREEZE_TAIL)

    # 5) Düşük LR ile derle
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    # 6) Class weight (green az → telafi)
    class_weights = compute_class_weights()

    # 7) Callback’ler
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(BEST_OUT, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    ]

    # (İstersen XLA/JIT uyarılarını azalt)
    # tf.config.optimizer.set_jit(False)

    print(f"[info] Fine-tune başlıyor… (unfreeze last ~{UNFREEZE_TAIL}, lr=1e-5)")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, class_weight=class_weights)

    print("Kaydedildi:", BEST_OUT)
