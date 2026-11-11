import json, numpy as np, tensorflow as tf
from config import IMG_SIZE, MODEL_PATH, CLASSES_PATH

model = tf.keras.models.load_model(MODEL_PATH)
class_names = json.load(open(CLASSES_PATH))

def predict_all(img):
    """Retourne la liste (categorie, probabilite) triée du plus au moins probable."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img), 0)
    probs = model.predict(arr, verbose=0)[0]
    results = [(class_names[i], float(p)) for i, p in enumerate(probs)]
    return sorted(results, key=lambda x: x[1], reverse=True)