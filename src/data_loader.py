import tensorflow as tf
from config import *

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.3, subset="training",
        seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.3, subset="validation",
        seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(AUTOTUNE),
            val_ds.prefetch(AUTOTUNE),
            class_names)
