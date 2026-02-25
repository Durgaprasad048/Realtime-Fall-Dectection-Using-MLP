import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EXTRA_SITE_PACKAGES = r"C:\tfpkgs"
if os.path.isdir(EXTRA_SITE_PACKAGES) and EXTRA_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, EXTRA_SITE_PACKAGES)

import numpy as np
try:
    import tensorflow as tf
except Exception as exc:
    raise SystemExit(
        "TensorFlow is unavailable or broken in this environment. "
        "Reinstall TensorFlow and retry.\n"
        f"Original error: {exc}"
    )

MODEL_DIR = "./model_x"
MODEL_FILE = "./model_x.keras"

if os.path.isfile(MODEL_FILE):
    model = tf.keras.models.load_model(MODEL_FILE)
else:
    try:
        model = tf.keras.models.load_model(MODEL_DIR)
    except ValueError:
        try:
            layer = tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
            inputs = tf.keras.Input(shape=(20, 6), dtype=tf.float32)
            outputs = layer(inputs)
            if isinstance(outputs, dict):
                outputs = next(iter(outputs.values()))
            model = tf.keras.Model(inputs, outputs)
        except Exception as exc:
            raise SystemExit(
                "Could not load model from './model_x' or './model_x.keras'. "
                "Run Model_Training.py to generate './model_x.keras'.\n"
                f"Original error: {exc}"
            )

print("Model loaded successfully")
model.summary()

sample_data = np.random.rand(1, 20, 6).astype(np.float32)
pred = model.predict(sample_data, verbose=0)
result = int(np.argmax(pred))

print("Predicted class id:", result)
