#%%
import tensorflow as tf
import os
def test():
    model = tf.saved_model.load("/mnt/d/pythonaijia/model/mnist/300")
    print(model)
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    (train, train_label), (test, test_label) =   tf.keras.datasets.cifar100.load_data()
    y_pred = model(test)
    sparse_categorical_accuracy.update_state(y_true=test_label,
                                             y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


test()
