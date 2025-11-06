import os, math
import tensorflow as tf
from pathlib import Path
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# --------------------
# Parametreler
# --------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_DIR = "data/dataset"
CLASSES = ["healthy","green","rotten"]
EPOCHS = 15

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = str(MODEL_DIR / "potato_model.keras")
BEST_MODEL_H5   = str(MODEL_DIR / "potato_model.h5")  # opsiyonel

# --------------------
# Dataset
# --------------------
def build_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        labels="inferred", label_mode="categorical",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        labels="inferred", label_mode="categorical",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    AUTOTUNE = tf.data.AUTOTUNE

    # Augmentation sadece train'de
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    def preprocess(x, y):
        # x: [0,255] RGB -> MobileNetV2 preprocess [-1,1]
        return preprocess_input(tf.cast(x, tf.float32)), y

    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.map(preprocess,   num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    return train_ds, val_ds

# --------------------
# Model
# --------------------
def build_model(num_classes=3, img_size=IMG_SIZE):
    base = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # frozen base

    inputs = layers.Input(shape=img_size + (3,))
    x = inputs
    # (preprocess zaten dataset'te uygulandı; burada tekrar etmiyoruz)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# --------------------
# Eğitim
# --------------------
def main():
    print("Dataset yükleniyor…")
    train_ds, val_ds = build_datasets()

    print("Model inşa ediliyor…")
    model = build_model(num_classes=len(CLASSES), img_size=IMG_SIZE)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    ]

    print("Eğitim başlıyor…")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # En iyi modeli yükle (ModelCheckpoint sayesinde BEST_MODEL_PATH yazıldı)
    best = tf.keras.models.load_model(BEST_MODEL_PATH)
    print("Val değerlendirme (best weights):")
    val_loss, val_acc = best.evaluate(val_ds, verbose=0)
    print(f"val_loss={val_loss:.4f}  val_accuracy={val_acc:.4f}")

    # Opsiyonel: .h5 olarak da kaydet (h5py yüklü)
    best.save(BEST_MODEL_H5)

    # Örnek: birkaç batch'ten toplu tahmin (hızlı bakış)
    for images, labels in val_ds.take(1):
        preds = best.predict(images, verbose=0)
        top = tf.argmax(preds, axis=1).numpy()
        true = tf.argmax(labels, axis=1).numpy()
        print("Örnek tahminler (ilk 10):")
        for i in range(min(10, len(top))):
            print(f"  true={CLASSES[true[i]]:7s}  pred={CLASSES[top[i]]:7s}  conf={float(np.max(preds[i])):.2f}")
        break

    print("Bitti. Modeller:")
    print(" -", BEST_MODEL_PATH)
    print(" -", BEST_MODEL_H5)

if __name__ == "__main__":
    main()
