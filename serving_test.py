#%%
import tensorflow as tf
import json
import requests
import numpy as np

(train, train_label), (test, test_label) = tf.keras.datasets.mnist.load_data()
# print(test[:1].tolist())
data = {'instances': [3,4]}
headers = {
    'Content-Type': 'application/json'
}
res = requests.post("http://localhost:8501/v1/models/prediction_of_housing_quantity:predict",json=data,headers=headers)
print(json.loads(res.text))