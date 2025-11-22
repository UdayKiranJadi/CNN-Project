# Image Classifier with TensorFlow & CNN


A small end‑to‑end image classification project:
- **Training** in Python/Keras on CIFAR‑10 (32×32×3).
- **Frontend** in React + **TensorFlow.js** to run the trained model in the browser.
- Optional Webcam inference and clean UI for upload + predictions.

## Abstract
We train a compact CNN with three convolutional blocks (Conv → BatchNorm → ReLU, ×2 per block), max‑pooling and dropout, followed by a dense head (256 → dropout → 10‑way softmax). The best weights (by validation accuracy) are exported to TF.js and used in a React app for real‑time inference. Final validation accuracy after fine‑tuning: ~**80–81%** on CIFAR‑10.

## Architecture (DL)
- Input: **32×32×3** RGB
- Block 1: Conv(3×3, 32) → BN → ReLU → Conv(3×3, 32) → ReLU → MaxPool(2×2) → Dropout(0.25)
- Block 2: Conv(3×3, 64) → BN → ReLU → Conv(3×3, 64) → ReLU → MaxPool(2×2) → Dropout(0.25)
- Block 3: Conv(3×3, 128) → BN → ReLU → Conv(3×3, 128) → ReLU → MaxPool(2×2) → Dropout(0.25)
- Head: Flatten (2048) → Dense(256, ReLU) → Dropout(0.5) → Dense(10, softmax)

Diagrams:
- `artifacts/cnn_block_diagram.png` – Block diagram
- `artifacts/cnn_technical.png` – Technical diagram with shapes
- `training/model/summary.txt` – Keras summary
- `training/model/architecture.png` – Keras plot
  - auto-created if `pydot` + `graphviz` installed

## Repo Structure
```
client/                    # React + Vite app (TF.js inference)
  public/model/            # TF.js model: model.json + group*-shard*.bin
  src/App.jsx              # UI + webcam + predictions
training/
  model_creation.py        # Train from scratch on CIFAR-10
  fine_tune.py             # Resume & fine‑tune from best checkpoint
  model/                   # Saved models (.keras / .h5) + labels.json
artifacts/                 # Diagrams exported for presentation.
```

## Quickstart (Frontend)
```bash
cd client
npm install
npm run dev
# Visit http://localhost:5173
```
Place the converted model in `client/public/model/` so the app can load `/model/model.json`.

## Train (Python/Keras)
Use a dedicated Conda env for training (CPU okay):
```bash
conda create -n tf-train310 python=3.10 -y
conda activate tf-train310
pip install tensorflow==2.19.* numpy matplotlib pydot graphviz
python training/model_creation.py
# Best Keras model will be saved to training/model/image_classifier_best.keras
```
Optional fine‑tune (resume from best):
```bash
python training/fine_tune.py --resume training/model/image_classifier_best.keras --epochs 20 --lr 1e-4
```

## Convert to TF.js
Use a **separate** Python 3.11 venv with TFJS converter 4.22 (H5 path recommended for compatibility):
```bash
# 1) In training env: export best .keras → .h5
python - <<'PY'
import tensorflow as tf
m = tf.keras.models.load_model('training/model/image_classifier_best.keras', compile=False)
m.save('training/model/image_classifier_best.h5')
print("Wrote training/model/image_classifier_best.h5")
PY

# 2) Activate TFJS env and convert
python -m pip install --upgrade pip
pip install tensorflowjs==4.22.0 h5py
mkdir -p client/public/model
tensorflowjs_converter   --input_format=keras   training/model/image_classifier_best.h5   client/public/model
# (Optional) Reduce size:
# tensorflowjs_converter --input_format=keras --quantize_float16 training/model/image_classifier_best.h5 client/public/model
```

Verify the files are served:
```bash
curl -I http://localhost:5173/model/model.json
# Expect: HTTP/1.1 200 OK and Content-Type: application/json
```

## Run Inference (UI)
- Upload a JPG/PNG or start webcam.
- The app preprocesses to **32×32** (contain/cover) and shows **Top‑2** predictions as percentages.
- Status banner indicates model loading / warm‑up / inference states.



## License
MIT (or institution policy).

## Acknowledgments
- CIFAR‑10 dataset (Krizhevsky et al.)
- TensorFlow/Keras and TensorFlow.js teams.
