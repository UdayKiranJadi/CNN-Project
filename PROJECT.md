# Project Description — Image Classifier with TensorFlow & CNN

## Objective
Build a compact CNN that classifies CIFAR-10 images and runs fully in the browser via TensorFlow.js, with a simple React UI for upload/webcam inference.

## Dataset
- **CIFAR-10** (60k images, 32×32 RGB, 10 classes)  
  Source: https://www.cs.toronto.edu/~kriz/cifar.html

## Method
- **Model:** 3 convolutional blocks (Conv→BatchNorm→ReLU ×2 per block), MaxPool(2×2), Dropout(0.25 each block).  
  Head: Flatten→Dense(256, ReLU)→Dropout(0.5)→Dense(10, Softmax).
- **Training:** TensorFlow/Keras (Python). Best checkpoint selected by validation accuracy.
- **Export:** Best Keras model → TF.js (via tensorflowjs_converter).
- **Frontend:** React + TF.js. Images are letterboxed to 32×32 (contain/cover). UI shows **Top-2** predictions with confidence.

## Results
- Validation accuracy after fine-tuning: **≈90–91%** on CIFAR-10.  
  (Fast, responsive in-browser inference; no server compute required.)

## How to Run
- **Train (Python):** `python training/model_creation.py` (see README for env setup)  
- **Convert:** TF.js converter → `client/public/model/`  
- **Frontend:** `cd client && npm install && npm run dev` → open `http://localhost:5173`

## Repository Layout
- `client/` React app (TF.js inference)  
- `training/` Keras training + fine-tune scripts and saved models  
- `artifacts/` Diagrams & model plots  
- `docs/announcement/` LaTeX + PDF announcement

## Diagrams
- `artifacts/cnn_block_diagram.png` (block diagram)  
- `artifacts/cnn_technical.png` (shapes/technical view)  
- `training/model/summary.txt`, `training/model/architecture.png` (Keras)


