# File: training/export_h5.py           # <- Python script: load .keras and save as .h5
import os                               # <- path handling
import tensorflow as tf                 # <- tf.keras loads/saves models

BASE = os.path.dirname(__file__)        # <- folder of this script ("training/")
src = os.path.join(BASE, "model", "image_classifier.keras")  # <- source .keras path
dst = os.path.join(BASE, "model", "image_classifier.h5")     # <- target .h5 path

print("Loading:", src)                  # <- log: show what we load
model = tf.keras.models.load_model(src) # <- load the .keras (Keras v3 single-file format)

print("Saving:", dst)                   # <- log: show where we save
model.save(dst)                         # <- saving with .h5 extension writes HDF5 using h5py

print("Done.")                          # <- finished
