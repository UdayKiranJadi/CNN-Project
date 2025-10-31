# File: training/model_creation.py
# Purpose: Train a small CNN on CIFAR-10 and save a Keras model for later TF.js conversion.

import os                                   # work with file paths
import json                                 # save class labels as JSON
import tensorflow as tf                     # TensorFlow (Keras included)
from tensorflow.keras import layers, models, callbacks  # Keras API parts

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)        # folder that contains this script
MODEL_DIR = os.path.join(BASE_DIR, "model") # folder to store the trained model files
os.makedirs(MODEL_DIR, exist_ok=True)       # create the folder if it doesn't exist

MODEL_PATH = os.path.join(MODEL_DIR, "image_classifier.keras")  # final Keras model path
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")            # class labels JSON path

# CIFAR-10 class names in index order
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def build_model():
    """Build and compile a small CNN for 32x32x3 images (CIFAR-10)."""
    model = models.Sequential([                 # FIX: models.Sequential (not model.Sequential)
        layers.Input(shape=(32, 32, 3)),        # input is 32x32 RGB
        layers.Rescaling(1.0 / 255.0),          # FIX: Rescaling (not Rescaleing) -> normalize to [0,1]

        # Block 1
        layers.Conv2D(32, 3, padding="same"),   # conv layer
        layers.BatchNormalization(),            # stabilize training
        layers.Activation("relu"),              # FIX: Activation (not Actvation)
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"), # add second conv for symmetry
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),                       # flatten feature maps
        layers.Dense(256, activation="relu"),   # dense layer
        layers.Dropout(0.5),                    # regularize
        layers.Dense(10, activation="softmax")  # 10 classes -> class probabilities
    ])

    model.compile(                              # compile with optimizer/loss/metrics
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    """Load CIFAR-10, train, and save model + labels."""
    # Optional: let TF use GPU memory gradually (safe if no GPU)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    print("[INFO] Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()                 # shape (N,)
    y_test = y_test.squeeze()

    print("[INFO] Building data pipeline...")
    aug = tf.keras.Sequential([                 # light augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])

    print("[INFO] Building model...")
    model = build_model()

    batch_size = 128                            # batch size
    epochs = 15                                 # number of epochs

    # tf.data pipelines for performance
    train_ds = (tf.data.Dataset
                .from_tensor_slices((x_train, y_train))
                .shuffle(50000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    train_ds = train_ds.map(lambda x, y: (aug(tf.cast(x, tf.float32)), y))

    val_ds = (tf.data.Dataset
              .from_tensor_slices((x_test, y_test))
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))

    cbs = [                                     # callbacks: save best + early stop
        callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    print("[INFO] Training...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=1)

    print("[INFO] Evaluating...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"[INFO] Validation accuracy: {val_acc:.4f}")

    print("[INFO] Saving labels...")
    with open(LABELS_PATH, "w") as f:
        json.dump(CLASS_NAMES, f)

    # Ensure model exists even if checkpoint didn’t trigger (edge case)
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Saving final model...")
        model.save(MODEL_PATH)

    print("[INFO] Done. Files saved to:", MODEL_DIR)

if __name__ == "__main__":                      # run main() only if script executed directly
    main()
