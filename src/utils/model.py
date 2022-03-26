import tensorflow as tf
import time
import os


def created_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

   
    LAYERS = [tf.keras.layers.Flatten(input_shape =[28,28]),
            tf.keras.layers.Dense(300, activation="relu"),
            tf.keras.layers.Dense(100, activation = "sigmoid"),
            tf.keras.layers.Dense(NUM_CLASSES, activation = "softmax")
        ] 
    model = tf.keras.models.Sequential(LAYERS)

    model.compile(optimizer=OPTIMIZER, loss= LOSS_FUNCTION, metrics=METRICS)

    model.summary()

    return model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S.h5_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)