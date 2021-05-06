import tensorflow as tf  # 导入模块
from sklearn import datasets  # 从sklearn中导入数据集
import numpy as np  # 导入科学计算模块
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

########################################################################################################
# AA=np.arange(0,15)
# frame_length=5
# num_frames=4
# frame_step=2
# A=np.arange(0, frame_length)
# print(A)
# indices0= np.tile(np.arange(0, frame_length), (num_frames, 1))
# print(indices0)
# indices1=np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
# print(indices1)
# indices = indices0+indices1+1
# print(indices)
# B=AA[indices.astype(np.int32)]
# print(AA)
# print(B)
# C1=np.arange(0,frame_length)
# C=B*C1
# print(C1)
# print(C)
# ########################################################################################################
# np.random.seed(19680801)
# Z = np.random.rand(6, 10)
# print(Z.shape)
# x = np.arange(-0.5, 10, 1)  # len = 11
# print(x.shape)
# y = np.arange(4.5, 11, 1)  # len = 7
# print(y.shape)
#
# fig, ax = plt.subplots()
# ax.pcolormesh(x, y, Z)
# plt.show()
########################################################################################################
# plt.subplot(212)
# plt.contourf(t, frequencies, abs(cwtmatr))
# plt.ylabel(u"频率(Hz)", fontproperties=chinese_font)
# plt.xlabel(u"时间(秒)", fontproperties=chinese_font)
# plt.subplots_adjust(hspace=0.4)
# plt.show()
########################################################################################################
# Z = np.tile(np.arange(0, 3), (3, 2))
# print(Z)
# print(Z[0, 1])
########################################################################################################
# BB=np.linspace(0,13,13)
# print(BB)
# A,B=np.mgrid[0:13+0.3*(13/12):13/12,0:5:1]
# C=np.random.randint(10,100,[13,5])
# print(A)
# print(B)
# print(C)
# plt.pcolormesh(A, B, C, cmap=plt.cm.cool)
# plt.colorbar()
# # plt.plot(np.arange(max_f.shape[0]), max_f)
# plt.ylabel('MFCCs')
# plt.xlabel('Time [sec]')
# plt.show()
########################################################################################################
# nrows = 3
# ncols = 5
# Z = np.arange(nrows * ncols).reshape(nrows, ncols)
# x = np.arange(ncols + 1)
# y = np.arange(nrows + 1)
#
# fig, ax = plt.subplots()
# ax.pcolormesh(x, y, Z, shading='flat', vmin=Z.min(), vmax=Z.max())
#
#
# def _annotate(ax, x, y, title):
#     # this all gets repeated below:
#     X, Y = np.meshgrid(x, y)
#     ax.plot(X.flat, Y.flat, 'o', color='m')
#     ax.set_xlim(-0.7, 5.2)
#     ax.set_ylim(-0.7, 3.2)
#     ax.set_title(title)
#
# _annotate(ax, x, y, "shading='flat'")
# plt.show()
########################################################################################################
# a = tf.constant([[[1,2,3]],[[4,5,6]]], tf.int32)
# print(a)
# print(a.shape)
# aa=tf.reshape(a, ())
# print(aa)
# print(aa.shape)
# b = tf.constant([1,2,1], tf.int32)
# c=tf.tile(a, b)
# print(c)
# print(c.shape)
#####################################################################################################
# name = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
#     'name', ['bob', 'george', 'wanda']))
# columns = [name, ...]
# features = tf.io.parse_example(..., features=tf.feature_column.make_parse_example_spec(columns))
# dense_tensor = tf.keras.layers.InputLayer(features, columns)
#
# dense_tensor == [[1, 0, 0]]  # If "name" bytes_list is ["bob"]
# dense_tensor == [[1, 0, 1]]  # If "name" bytes_list is ["bob", "wanda"]
# dense_tensor == [[2, 0, 0]]  # If "name" bytes_list is ["bob", "bob"]
#####################################################################################################

# ar1 = np.array(['Red', 'Blue', 'Green', 'Orange'])
# ar2 = np.array(['Black', 'Yellow'])
#
# # Concatenate array ar1 & ar2
# ar3 = np.concatenate(ar1, ar1)
# print(ar3)

# ar1 = np.array(['Red', 'Blue', 'Green', 'Orange'])
# ar2 = np.array(['Black', 'Yellow'])
#
# # Concatenate array ar1& ar2 by List
# ar3 = np.concatenate([ar1, ar1])
# print(ar3)
# print(ar3[:-1])
# print(ar3[0:-1])
# print(ar3[-1:])
# print(ar3[-1])
#####################################################################################################

# X_train = np.random.random((14125,18))
# y_train = np.random.random((14125,4))
#
# dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_data = dataset.shuffle(len(X_train)).batch(32)
# train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
# tfmodel = tf.keras.Sequential([
#                   tf.keras.layers.Dense(15, activation=tf.nn.relu, input_shape=(18,)),
#                   # tf.keras.layers.Flatten(),
#                   tf.keras.layers.Dense(10, activation=tf.nn.relu),
#                   tf.keras.layers.Dense(4, activation=tf.nn.softmax)
# ])
#
# tfmodel.compile(optimizer='adam',
#                 loss=tf.keras.losses.CategoricalCrossentropy(),
#                 metrics=['accuracy'])
#
# tfmodel.fit(train_data, epochs=5)
#####################################################################################################
# m = tf.keras.metrics.SparseCategoricalAccuracy()
# # A=[3, 2, 1, 0, 0, 1, 2, 3]
# A=[4, 3, 2, 1, 1, 2, 3, 4]
# B=[[0.9, 0.1, 0.1, 0.1],[0.1, 0.1, 0.9, 0.1],[0.9, 0.1, 0.1, 0.1],[0.9, 0.1, 0.1, 0.1],
#    [0.9, 0.1, 0.1, 0.1],[0.9, 0.1, 0.1, 0.1],[0.9, 0.1, 0.1, 0.1],[0.9, 0.1, 0.1, 0.1]]
# m.update_state(A,B)
# print(m.result().numpy())
#####################################################################################################
box_p=tf.random.uniform(shape=[12],minval=0, maxval=1,dtype=tf.float32)
print('box_p',box_p)
box_p=box_p.numpy()
order = box_p.argsort()[::-1]
print('order.size',order.size)
print('order',order)
xx1 = np.minimum(box_p[order[0]], box_p[order[1:]])
print(xx1,xx1.shape)
inds = np.where(xx1 <= 0.5)
print(inds,inds[0])
print(order[inds[0]+1])
print(order)
print(box_p[order[inds[0]+1]])

