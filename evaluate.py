import os, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.mobilenet_v2 import preprocess_input

DATA_DIR   = "data/dataset"
CLASSES    = ["healthy","green","rotten"]
BATCH_SIZE = 32

def pick_model_path():
    ft = "models/potato_model_ft.keras"
    base = "models/potato_model.keras"
    return ft if os.path.exists(ft) else base

def build_val_ds(img_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        labels="inferred", label_mode="categorical",
        image_size=img_size, batch_size=BATCH_SIZE, shuffle=False
    )
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(lambda x,y: (preprocess_input(tf.cast(x, tf.float32)), y),
                num_parallel_calls=AUTOTUNE)
    return ds

if __name__ == "__main__":
    model_path = pick_model_path()
    print("Model:", model_path)
    model = tf.keras.models.load_model(model_path)

    # Modelin beklediği giriş boyutu (H, W) → (None, H, W, 3)
    H, W = model.input_shape[1], model.input_shape[2]
    IMG_SIZE = (H, W)
    print("IMG_SIZE:", IMG_SIZE)

    val_ds = build_val_ds(IMG_SIZE)

    y_true, y_pred = [], []
    for imgs, labels in val_ds:
        preds = model.predict(imgs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true); y_pred = np.array(y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    print("Confusion Matrix (rows: true, cols: pred):")
    print(confusion_matrix(y_true, y_pred))
