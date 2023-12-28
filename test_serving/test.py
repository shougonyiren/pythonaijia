#%%
import tensorflow as tf
import tensorflow

from tensorflow import keras

from tensorflow import keras as kr
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV3Small
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    num_epochs = 1
    batch_size = 32
    learning_rate = 0.001

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation=tf.nn.relu),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Softmax()
    ])

    (train, train_label), (test, test_label) = \
        tf.keras.datasets.cifar100.load_data()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(train, train_label, epochs=num_epochs, batch_size=batch_size)
    print("tf.saved_model.save")
    tf.saved_model.save(model, "/mnt/d/pythonaijia/model/mnist/300")

main()