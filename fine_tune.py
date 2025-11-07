import os
import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_DIR = "data/dataset"
CLASSES = ["healthy","green","rotten"]
EPOCHS = 10

BEST_MODEL_PATH = "models/potato_model.keras"
FT_MODEL_PATH   = "models/potato_model_ft.keras"

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

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    def preprocess(x, y):
        return preprocess_input(tf.cast(x, tf.float32)), y

    train_ds = train_ds.map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.map(preprocess,   num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    return train_ds, val_ds

def main():
    train_ds, val_ds = build_datasets()

    # 1) Frozen modelden başla
    base_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    # base_model.layers[1] MobilenetV2 gövdesi (functional) olabilir; referans bulalım:
    mobilenet = None
    for l in base_model.layers:
        if isinstance(l, tf.keras.Model) and "mobilenetv2" in l.name:
            mobilenet = l
            break
    if mobilenet is None:
        # Alternatif: gövdeyi yeniden kur ve tepeyi base_model'den al
        mobilenet = MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights="imagenet")

    # 2) Son katmanları aç: ~30 layer iyi başlangaç
    unfreeze_from = max(0, len(mobilenet.layers) - 30)
    for i, l in enumerate(mobilenet.layers):
        l.trainable = (i >= unfreeze_from)

    # 3) Tüm modeli yeniden derle — küçük LR
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(FT_MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    ]

    print("Fine-tuning başlıyor… (son ~30 katman açık, lr=1e-5)")
    base_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # En iyi ağırlıklarla kaydedildi
    print("Fine-tuning bitti. Kaydedilen:", FT_MODEL_PATH)

if __name__ == "__main__":
    main()
