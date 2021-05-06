import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
###########################################################
###########################################################
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
pd.set_option('display.max_rows', None)#WebSite'pandas中关于DataFrame行，列显示不完全（省略）的解决办法 https://blog.csdn.net/weekdawn/article/details/81389865'
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_columns', 10)
dataframe = pd.read_csv(URL)
print(dataframe.head())
train, test = train_test_split(dataframe, test_size=0.2) #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 5 # 小批量大小用于演示
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)