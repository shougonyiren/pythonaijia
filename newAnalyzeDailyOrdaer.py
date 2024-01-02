

#%%


import pathlib

import pathlib

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

import tensorflow as tf
from keras.models import load_model
import keras
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.keras.models import load_model

wsl_path = "/mnt/z/爱家工作/项目py数据/count.xlsx"
# myfont = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")
# sns.set(font=myfont.get_name())
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

print(tf.__version__)

#订单创建日期  房源id  拍摄数量  事业部数量  weekday   星期  index
column_names = ["订单创建日期", "房源id","拍摄数量","事业部数量",  "weekday", "星期" ,"index"]

raw_dataset = pd.read_excel(wsl_path, names=column_names,
                      # na_values = "?", comment='\t',
                          # encoding='ISO-8859-1',
                          # encoding_errors = 'ignore',
                      # sep=" ", skipinitialspace=True
                            )

dataset = raw_dataset.copy()
dataset.pop("房源id")
dataset.pop("订单创建日期")
dataset.pop("星期")

dataset.tail()
dataset.isna().sum()


dataset = dataset.dropna()
dataset.set_index("index")

print(dataset.tail())

# print(dataset.isna().sum())

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# train_dataset = dataset.sample(frac=0.8,random_state=0)
# test_dataset = dataset.drop(train_dataset.index)

print("train_dataset")
print(train_dataset)

# "订单创建日期","拍摄数量",
sns.pairplot(train_dataset[["拍摄数量","事业部数量","weekday","index"]], diag_kind="kde")

plt.show()

train_stats = train_dataset.describe()
train_stats.pop("拍摄数量")
train_stats = train_stats.transpose()
print("train_stats")
print(train_stats)

train_labels = train_dataset.pop("拍摄数量")
test_labels = test_dataset.pop("拍摄数量")

print("train_labels")
print(train_labels)

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print("normed_train_data")
print(normed_train_data)

normed_train_data=normed_train_data.astype('float64')
normed_test_data=normed_test_data.astype('float64')
print("normed_train_data1")
print(normed_train_data)

def build_model():
  # print("len(train_dataset.keys())")
  # print(len(train_dataset.keys()))
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


model = build_model()

model.summary()


print('normed_train_data[index]')

print(normed_train_data.columns)

# normed_train_data.drop("拍摄数量")
# normed_train_data=normed_train_data[["index","weekday","事业部数量"]]
# del normed_train_data["拍摄数量"]
# del normed_test_data["拍摄数量"]
print(normed_train_data)

example_batch = normed_train_data[:10]
print('example_batch')
print(example_batch)


# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,100])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20000])
  plt.legend()
  plt.show()


plot_history(history)
# print(history.history)


model = build_model()

# patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
# print(history.history)


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))



test_predictions = model.predict(normed_test_data).flatten()

print(test_predictions)
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [拍摄量]')
plt.ylabel('Predictions [拍摄量]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()


# 误差分布
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [拍摄数量]")
_ = plt.ylabel("数量")
plt.show()



#%%
# 预测模型
import numpy as np

# Create and populate a 5x2 NumPy array.
my_data =np.array([[36,0,192],])

# Create a Python list that holds the names of the two columns.
my_column_names = ["事业部数量","weekday","index"]

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)
normed_data = norm(my_dataframe)
# print(model.predict(normed_test_data))
my_test_predictions = model.predict(normed_data).flatten()
print(my_test_predictions)


#%%
# 保存模型
# tf.saved_model.save(model,export_dir='/mnt/d/pythonaijia/model/mnist/411')
# model.save('/mnt/d/pythonaijia/model/mnist/128')
# 模型保存，注意：仅仅是多了一个save_format的参数而已
# 注意：这里的'path_to_saved_model'不再是模型名称，仅仅是一个文件夹，模型会保存在这个文件夹之下
model.save('/mnt/d/pythonaijia/model/mnist/412')

# 加载模型，通过指定存放模型的文件夹来加载
# model = tf.saved_model.load("/mnt/d/pythonaijia/model/mnist/411")

# new_model = keras.models.load_model('path_to_saved_model')
#%%
# import tensorflowjs as tfjs
# from keras.layers import Layer, InputSpec
# tfjs_target_dir ="/mnt/d/pythonaijia/tfjs_target"
# tfjs.converters.save_keras_model(model, tfjs_target_dir)



#%%
# 爱家工作/项目py数据/count.xlsx
# raw_dataset = pd.read_excel("/mnt/z/爱家工作/项目py数据/count.xlsx", names=column_names,
#                       # na_values = "?", comment='\t',
#                           # encoding='ISO-8859-1',
#                           # encoding_errors = 'ignore',
#                       # sep=" ", skipinitialspace=True
#                             )



