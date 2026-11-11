import json, numpy as np, tensorflow as tf
from data_loader import load_datasets
from model import build_model
from config import MODEL_PATH, CLASSES_PATH

train_ds, val_ds, class_names = load_datasets()
model, base = build_model(len(class_names))

# Poids de classes : les catégories avec moins d'images comptent plus
labels = np.concatenate([y.numpy() for _, y in train_ds])
counts = np.bincount(labels, minlength=len(class_names))
class_weight = {i: counts.sum() / (len(class_names) * c) for i, c in enumerate(counts)}
print("Poids des classes :", class_weight)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
]

# Phase 1 : entraînement de la tête
model.fit(train_ds, validation_data=val_ds, epochs=30,
          callbacks=callbacks, class_weight=class_weight)

# Phase 2 : fine-tuning
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=30,
          callbacks=callbacks, class_weight=class_weight)

json.dump(class_names, open(CLASSES_PATH, "w"))