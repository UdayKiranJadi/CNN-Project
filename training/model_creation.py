

import os
import json
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)                  # folder containing this script
MODEL_DIR = os.path.join(BASE_DIR, "model")           # output folder
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "image_classifier.keras")
BEST_PATH = os.path.join(MODEL_DIR, "image_classifier_best.keras")
MANIFEST_PATH = os.path.join(MODEL_DIR, "best_manifest.json")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# CIFAR-10 class names in index order
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

SEED = 1337
BATCH_SIZE = 128
EPOCHS = 40  # EarlyStopping will stop earlier if no improvement


def set_memory_growth() -> None:
    """Allow TF to allocate GPU memory on-demand (safe no-op if no GPU)."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def set_seed(seed: int = 1337) -> None:
    """Set global seeds for reproducibility (as much as practical)."""
    tf.keras.utils.set_random_seed(seed)


def build_model() -> tf.keras.Model:
    """Build and compile a small 32x32 CNN with an in-graph 1/255 Rescaling layer."""
    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Rescaling(1.0 / 255.0, name="rescaling"),

            # Block 1
            layers.Conv2D(32, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, 3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_datasets(batch_size: int = 128, seed: int = 1337):
    """Create train/validation tf.data pipelines with light augmentation on training."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    aug = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augment",
    )

    def map_train(x, y):
        x = tf.cast(x, tf.float32)         # model will rescale 1/255
        x = aug(x, training=True)
        return x, y

    def map_val(x, y):
        x = tf.cast(x, tf.float32)
        return x, y

    autotune = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=50000, seed=seed, reshuffle_each_iteration=True)
        .map(map_train, num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(map_val, num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )

    return train_ds, val_ds


def train_model(model: tf.keras.Model, train_ds, val_ds, epochs: int, model_path: str):
    """Train the model with callbacks: save-best, early-stopping, and LR reduction."""
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs,
        verbose=1,
    )
    return history


def evaluate_model(model: tf.keras.Model, val_ds) -> float:
    """Evaluate the (restored-best) model on validation set and return accuracy."""
    _, val_acc = model.evaluate(val_ds, verbose=0)
    return float(val_acc)


def save_labels(labels_path: str, class_names) -> None:
    """Write class label names to JSON for reference/conversion."""
    with open(labels_path, "w") as f:
        json.dump(list(class_names), f, indent=2)


def ensure_model_exists(model: tf.keras.Model, model_path: str) -> None:
    """If checkpoint did not trigger (edge case), save the current model snapshot."""
    if not os.path.exists(model_path):
        model.save(model_path)


def keep_best_across_runs(
    this_run_best_path: str,
    global_best_path: str,
    manifest_path: str,
    history: tf.keras.callbacks.History,
    fallback_val_acc: float,
) -> None:
    """
    Maintain a persistent 'best across runs' copy using val_accuracy.

    - Reads previous best score from MANIFEST (if any).
    - Gets this run's best val_accuracy from history (fallback to evaluate()).
    - If improved, copies this run's best model to global_best_path and updates MANIFEST.
    """
    # Prefer the best val_accuracy seen during this run
    try:
        val_hist = history.history.get("val_accuracy", [])
        this_best = float(max(val_hist)) if val_hist else float(fallback_val_acc)
    except Exception:
        this_best = float(fallback_val_acc)

    prev = {"score": -1.0}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                prev = json.load(f)
        except Exception:
            prev = {"score": -1.0}

    prev_score = float(prev.get("score", -1.0))

    if os.path.exists(this_run_best_path) and this_best > prev_score:
        shutil.copy2(this_run_best_path, global_best_path)
        with open(manifest_path, "w") as f:
            json.dump(
                {"metric": "val_accuracy", "score": this_best, "path": global_best_path},
                f,
                indent=2,
            )
        print(f"[BEST] New global best saved → {global_best_path} (val_acc={this_best:.4f})")
    else:
        print(f"[BEST] Kept previous global best (val_acc={prev_score:.4f})")


def main() -> None:
    """Orchestrate: seeds, data, model, training, evaluation, and saving artifacts."""
    set_memory_growth()
    set_seed(SEED)

    print("[INFO] Building datasets…")
    train_ds, val_ds = build_datasets(batch_size=BATCH_SIZE, seed=SEED)

    print("[INFO] Building model…")
    model = build_model()
    model.summary()

    print("[INFO] Training…")
    history = train_model(model, train_ds, val_ds, epochs=EPOCHS, model_path=MODEL_PATH)

    print("[INFO] Evaluating best weights…")
    val_acc = evaluate_model(model, val_ds)
    print(f"[INFO] Validation accuracy: {val_acc:.4f}")

    print("[INFO] Saving labels…")
    save_labels(LABELS_PATH, CLASS_NAMES)

    # Ensure a model exists on disk even if checkpoint didn't fire (rare)
    ensure_model_exists(model, MODEL_PATH)

    # Keep best-across-runs model beside the per-run best
    keep_best_across_runs(
        this_run_best_path=MODEL_PATH,
        global_best_path=BEST_PATH,
        manifest_path=MANIFEST_PATH,
        history=history,
        fallback_val_acc=val_acc,
    )

    print("[INFO] Done. Files saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
