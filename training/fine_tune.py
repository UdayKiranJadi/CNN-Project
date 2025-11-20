# training/fine_tune.py
# Purpose: Fine-tune CIFAR-10 model with AdamW+EMA, stronger augmentation, label smoothing.

import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
BEST_PATH = os.path.join(MODEL_DIR, "image_classifier_best.keras")

def build_model(lr: float):
    """Builds a CIFAR-10 CNN (32×32×3) and compiles it with AdamW + EMA."""
    m = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Rescaling(1/255.0),

        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, use_ema=True, ema_momentum=0.999)
    m.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
              metrics=["accuracy"])
    return m

def make_datasets(batch=128):
    """Creates tf.data pipelines for CIFAR-10 with stronger augmentation and one-hot labels."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze(); y_test = y_test.squeeze()

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.20),
        layers.RandomTranslation(0.125, 0.125, fill_mode="reflect"),
    ])

    def map_train(x, y):
        x = tf.cast(aug(x), tf.float32)       # keep values in [0,255], model rescales inside
        y = tf.one_hot(y, 10)                 # needed for label smoothing
        return x, y

    def map_val(x, y):
        x = tf.cast(x, tf.float32)
        y = tf.one_hot(y, 10)
        return x, y

    train = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
             .shuffle(50_000).batch(batch)
             .map(map_train, num_parallel_calls=tf.data.AUTOTUNE)
             .prefetch(tf.data.AUTOTUNE))

    val = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .batch(batch)
           .map(map_val, num_parallel_calls=tf.data.AUTOTUNE)
           .prefetch(tf.data.AUTOTUNE))
    return train, val

def load_or_build(resume_path: str | None, lr: float):
    """Loads an existing .keras (compile=False) and recompiles with AdamW+EMA; else builds fresh."""
    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] Resuming from: {resume_path}")
        m = tf.keras.models.load_model(resume_path, compile=False)
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, use_ema=True, ema_momentum=0.999)
        m.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=["accuracy"])
        return m
    print("[INFO] Building a fresh model")
    return build_model(lr)

def main():
    """Parses args, trains with callbacks, saves best as image_classifier_best.keras."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default="", help="Path to existing .keras to resume from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    for gpu in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except Exception: pass

    train_ds, val_ds = make_datasets(batch=args.batch)
    model = load_or_build(args.resume, args.lr)

    cbs = [
        callbacks.ModelCheckpoint(BEST_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, "tb_logs")),
    ]

    print("[INFO] Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs, verbose=1)

    print("[INFO] Evaluating best weights...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"[INFO] Val accuracy: {val_acc:.4f}")
    print(f"[INFO] Best model saved to: {BEST_PATH}")

if __name__ == "__main__":
    main()
