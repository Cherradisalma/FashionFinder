import json, numpy as np, tensorflow as tf
from config import IMG_SIZE, MODEL_PATH, CLASSES_PATH

model = tf.keras.models.load_model(MODEL_PATH)
class_names = json.load(open(CLASSES_PATH))

def predict(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img), 0)
    probs = model.predict(arr, verbose=0)[0]
    i = int(np.argmax(probs))
    return class_names[i], float(probs[i])
