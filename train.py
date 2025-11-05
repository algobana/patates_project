import os
import tensorflow as tf
from pathlib import Path

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_DIR = "data/dataset"
CLASSES = ["healthy","green","rotten"]

def build_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        labels="inferred", label_mode="categorical",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        labels="inferred", label_mode="categororical" if False else "categorical",
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    return train_ds, val_ds

if __name__ == "__main__":
    print("Dataset kontrol ediliyor…")
    train_ds, val_ds = build_datasets()
    print("Sınıflar:", CLASSES)
    print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Hazır. (Eğitim Adım 2’de)")
