import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

##########tf.io.gfile.glob equals glob
import struct
import os
import matplotlib.pyplot as plt
# 使用tfrecord的原因:正常情况下我们训练文件夹经常会生成 train, test 或者val文件夹，这些文件夹内部往往会存着成千上万的图片或文本等文件
# 这些文件被散列存着，这样不仅占用磁盘空间，并且再被一个个读取的时候会非常慢，繁琐。占用大量内存空间（有的大型数据不足以一次性加载）。
# 此时我们TFRecord格式的文件存储形式会很合理的帮我们存储数据。
# TFRecord内部使用了“Protocol Buffer”二进制数据编码方案，它只占用一个内存块，只需要一次性加载一个二进制文件的方式即可，简单，快速，尤其对大型训练数据很友好。
# 而且当我们的训练数据量比较大的时候，可以将数据分成多个TFRecord文件，来提高处理效率。
###############################################
def _parse_image_function(example_proto):
    data_feature_description = {
        'efid_index_cycle': tf.io.FixedLenFeature([], tf.int64),# shape [] means scalar
        'label': tf.io.FixedLenFeature([], tf.int64),
        'wave_data': tf.io.FixedLenFeature(24, tf.float32),
        # 'wave_data': tf.io.VarLenFeature(tf.float32),
    }
  # Parse the input tf.train.Example proto using the dictionary above.
  #   return tf.io.parse_single_example(example_proto, data_feature_description)
  #   return tf.io.parse_example(example_proto, data_feature_description)

    features=tf.io.parse_example(example_proto, data_feature_description)
    # features = tf.io.parse_single_example(example_proto, data_feature_description)
    wave = features['wave_data']
    label = tf.cast(features['label'], tf.int32)
    wave = tf.reshape(wave, [1,24])
    # label = tf.reshape(label, [1])
    # efid = tf.cast(features['efid_index_cycle'], tf.int32)
    return wave,label


def _parse_image_function_wave(example_proto):
    data_feature_description = {
        'efid_index_cycle': tf.io.FixedLenFeature([], tf.int64),# shape [] means scalar
        'wave_data': tf.io.FixedLenFeature(24, tf.float32),
    }


    features=tf.io.parse_example(example_proto, data_feature_description)
    wave = features['wave_data']
    wave = tf.reshape(wave, [1,24])
    efid = tf.cast(features['efid_index_cycle'], tf.int32)

    return wave,efid

def _parse_image_function_label(example_proto):
    data_feature_description = {
        'efid_index_cycle': tf.io.FixedLenFeature([], tf.int64),# shape [] means scalar
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    features=tf.io.parse_example(example_proto, data_feature_description)
    label = tf.cast(features['label'], tf.int32)
    efid = tf.cast(features['efid_index_cycle'], tf.int32)

    return label,efid

def Plot_history(history):
    plt.plot(history.history['sparse_categorical_accuracy'], label='accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 0.8])
    plt.legend(loc='lower right')
    plt.show()

###############################################
def model_VGG(shape,fil_output):
    inputs = tf.keras.Input(shape=shape) #Input() is used to instantiate a Keras tensor.
    # inputs = tf.keras.layers.InputLayer(shape=shape)#Layer to be used as an entry point into a Network (a graph of layers).
    vgg_inputs = tf.keras.layers.Conv1D(fil_output, 3, activation='relu', padding='same')(inputs)
    vgg_inputs = tf.keras.layers.Conv1D(fil_output, 3, activation='relu', padding='same')(vgg_inputs)
    vgg_inputs = tf.keras.layers.MaxPooling1D(2, 1, padding='same')(vgg_inputs)

    vgg_inputs = tf.keras.layers.Conv1D(fil_output * 2, 3, activation='relu', padding='same')(vgg_inputs)
    vgg_inputs = tf.keras.layers.Conv1D(fil_output * 2, 3, activation='relu', padding='same')(vgg_inputs)
    vgg_inputs = tf.keras.layers.MaxPooling1D(2, 1, padding='same')(vgg_inputs)
    #
    #
    # vgg_inputs = tf.keras.layers.Conv1D(fil_output * 4, 3, activation='relu', padding='same')(vgg_inputs)
    # vgg_inputs = tf.keras.layers.Conv1D(fil_output * 4, 3, activation='relu', padding='same')(vgg_inputs)
    # vgg_inputs = tf.keras.layers.MaxPooling1D(2, 1, padding='same')(vgg_inputs)
    #
    # vgg_inputs = tf.keras.layers.Conv1D(fil_output * 8, 3, activation='relu', padding='same')(vgg_inputs)
    # vgg_inputs = tf.keras.layers.Conv1D(fil_output * 8, 3, activation='relu', padding='same')(vgg_inputs)
    # vgg_inputs = tf.keras.layers.MaxPooling1D(2, 1, padding='same')(vgg_inputs)

    # vgg_inputs = layers.Flatten()(vgg_inputs)
    vgg_inputs = tf.keras.layers.Dense(128, activation="relu")(vgg_inputs)
    vgg_inputs = tf.keras.layers.Dropout(0.1)(vgg_inputs)
    vgg_inputs = tf.keras.layers.Dense(64, activation="relu")(vgg_inputs)
    vgg_inputs = tf.keras.layers.Dropout(0.1)(vgg_inputs)
    # vgg_inputs = tf.keras.layers.Dense(32, activation="relu")(vgg_inputs)

    vgg_inputs = tf.keras.layers.Dense(4)(vgg_inputs)
    vgg_inputs = tf.keras.layers.Softmax()(vgg_inputs)
    model = tf.keras.Model(inputs=inputs, outputs=vgg_inputs)

    return model

def model_mnist_fashion_newver(shape):
    inputs = tf.keras.Input(shape=shape)
    vgg_inputs=tf.keras.layers.Dense(128, activation='relu')(inputs)
    # model.add(tf.keras.layers.Flatten())
    vgg_inputs=tf.keras.layers.Dense(64, activation='relu')(vgg_inputs)
    vgg_inputs=tf.keras.layers.Dense(4)(vgg_inputs)
    vgg_inputs = tf.keras.layers.Softmax()(vgg_inputs)

    model = tf.keras.Model(inputs=inputs, outputs=vgg_inputs)

    return model


def model_mnist_fashion_fast_test(shape):
  # model = tf.keras.Sequential([
  #   tf.keras.layers.Flatten(input_shape=(28, 28)),
  #   tf.keras.layers.Dense(128, activation='relu'),
  #   tf.keras.layers.Dense(10)
  #
  # ])
  # model = tf.keras.models.Sequential()
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(128, activation='relu',input_shape=shape))
  # model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(4))

  return model

def ConfusionMatrix_label_predictor(test_dataset):
    label_predict = []
    label_real = []
    count = 0
    for features in test_dataset:
        real_label = features[1]
        if real_label == 0:
            count = count + 1
        # label_real.append(real_label.numpy().reshape(1,-1).tolist())
        # label_real.append(real_label.numpy().reshape(1, -1))
        label_real.append(real_label.numpy())

        wave = features[0]
        prediction = np.argmax(model.predict(wave))
        # label_predict.append(prediction.reshape(1,-1).tolist())
        # label_predict.append(prediction.reshape(1, -1))
        label_predict.append(prediction)

    label_predict = np.asarray(label_predict)
    label_real = np.asarray(label_real).reshape(-1)
    # print(label_predict.shape)
    # print(label_real.shape)
    print(count)
    '''The matrix columns represent the prediction labels and the rows represent the real labels.
    Class labels are expected to start at 0'''
    CM=tf.math.confusion_matrix(labels=label_real,  predictions=label_predict, num_classes=4)
    CM=tf.cast(CM, tf.float32)
    return CM

def label_calculator(test_dataset):
    label_real = []
    for features in test_dataset:
        real_label = features[1]
        label_real.append(real_label.numpy())
    label_real = np.asarray(label_real).reshape(-1)
    CM=tf.math.confusion_matrix(labels=label_real,  predictions=label_real, num_classes=4)
    B = tf.reduce_sum(CM, axis=1)

    return B

###############################################



if __name__ == '__main__':
    # fsns_test_file = 'ICBHI_processed_data/Wave_MFCCs.tfrecords'
    # fsns_test_file = 'ICBHI_processed_data/Wave_MFCCs_more_detail.tfrecords'
    # fsns_test_file = 'ICBHI_processed_data/Wave.tfrecords'
    # raw_dataset = tf.data.TFRecordDataset(filenames=[fsns_test_file])
    Traning_file = 'ICBHI_processed_data/Wave_MFCCs_training.tfrecords'
    Test_file = 'ICBHI_processed_data/Wave_MFCCs_testing.tfrecords'
    raw_dataset_traning = tf.data.TFRecordDataset(filenames=[Traning_file])
    raw_dataset_testing = tf.data.TFRecordDataset(filenames=[Test_file])


    # print(raw_dataset.element_spec)
    # print(raw_dataset)
    # for raw_record in raw_dataset.take(5):
    #     print(repr(raw_record))
    #     parsed = tf.train.Example.FromString(raw_record.numpy())
    #     print('parsed=',parsed)
    #     print('only output label=', parsed.features.feature['label'])
    ###############################################  another way:
    # '''要解码消息，请使用 tf.train.Example.FromString() 方法。'''
    # for element in raw_dataset.as_numpy_iterator():
    #   parsed_element = tf.train.Example.FromString(element)
    #   print('The number of {} samples label is {}'.format(parsed_element.features.feature['efid_index_cycle'],parsed_element.features.feature['label']))
    ###############################################
    ################ inspec raw data set
    # parsed_raw_dataset= raw_dataset.map(_parse_image_function)
    # print(parsed_raw_dataset.element_spec)
    parsed_dataset_traning = raw_dataset_traning.map(_parse_image_function)
    parsed_dataset_testing = raw_dataset_testing.map(_parse_image_function)
    Traning_data_inspec=label_calculator(parsed_dataset_traning)
    print(Traning_data_inspec)

    # parsed_raw_dataset_label= raw_dataset.map(_parse_image_function_label)
    # parsed_raw_dataset_wave = raw_dataset.map(_parse_image_function_wave)
    # print(parsed_raw_dataset_label.element_spec)
    # print(parsed_raw_dataset_wave.element_spec)
    '''dataset API for pipeline, but Y target integer not support dataset format, 
    can only be tensor or numpy'''


    # print(parsed_raw_dataset)
    # for raw_record in parsed_raw_dataset.take(10):
    #     print(repr(raw_record))
    #

    ############################################################################
    '''map方法可以接受任意函数以对dataset中的数据进行处理；
    另外，可使用repeat、shuffle、batch方法对dataset进行重复、混洗、分批；用repeat复制dataset以进行多个epoch；如下：
    dataset = dataset.repeat(epochs).shuffle(buffer_size).batch(batch_size)
    dataset_train = parsed_raw_dataset.repeat(1).shuffle(3000).batch(1)
    # make sure repeat is ahead batch
    # this is different from dataset.shuffle(1000).batch(batch_size).repeat(epochs)
    # the latter means that there will be a batch data with nums less than batch_size for each epoch
    # if when batch_size can't be divided by nDatas.'''

    ############################################################################
    # DATASET_SIZE=6897
    # val_size = int(0.15 * DATASET_SIZE)
    # test_size = int(0.15 * DATASET_SIZE)
    # train_size = DATASET_SIZE-val_size-test_size
    #
    # full_dataset = parsed_raw_dataset.shuffle(DATASET_SIZE)
    # # full_dataset = parsed_raw_dataset
    # train_dataset = full_dataset.take(train_size)
    # test_dataset = full_dataset.skip(train_size)
    #
    # val_dataset =  test_dataset.skip(val_size)
    # test_dataset = test_dataset.take(test_size)

    DATASET_SIZE=6898
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)
    train_size = DATASET_SIZE-val_size-test_size

    train_dataset = parsed_dataset_traning.shuffle(train_size)
    test_dataset = parsed_dataset_testing.shuffle(train_size)

    val_dataset =  test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    ############################################################################
    # count = 0
    # for features in train_dataset:
    #     count = count + 1
    #     # image_features1 = features[1]
    #     # image_features2 = features[0]
    #     # print(image_features1.shape)
    #     # print(image_features2.shape)
    #     # print(image_features1)
    #     # print(image_features2)
    #     print(features)
    # print(count)

    # for features in test_dataset:
    #     count = count + 1
    #     image_features1 = features[1]
    #     image_features2 = features[0]
    #     # print(image_features1.shape)
    #     # print(image_features2.shape)
    #     print(image_features1)
    #     # print(image_features2)
    #     # print(features)
    # print(count)

    train_dataset=train_dataset.repeat(1)
    train_dataset=train_dataset.batch(5)

    val_dataset =  val_dataset.batch(1)
    test_dataset = test_dataset.batch(1)


    Input_layer_shape = (1,24)
    #############################################################
    # model = model_mnist_fashion_newver(Input_layer_shape)
    model = model_VGG(Input_layer_shape, 48)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  # metrics=['accuracy'])
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


    model.summary()  # # 显示模型的结构

    checkpoint_path = "model_weight_saving/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,  monitor='val_loss',save_weights_only=True,
                                                     save_best_only=True, verbose=1)

    history =model.fit(train_dataset, epochs=30, shuffle=False, validation_data=val_dataset, callbacks=cp_callback)
    # history = model.fit(train_dataset, epochs=3, shuffle=False, validation_data=val_dataset)

    Plot_history(history)
    #############################################################
    # #############################################################
    # '''然后从 checkpoint 加载权重并重新评估：
    # 加载权重'''
    # checkpoint_path = "model_weight_saving/cp.ckpt"
    # model = model_VGG(Input_layer_shape, 48)
    # # model = model_mnist_fashion_newver(Input_layer_shape)
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #               # metrics=['accuracy'])
    #               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    #
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # model.load_weights(checkpoint_path)
    #
    # #############################################################
    print('################# Start Evalulating: ')
    eva_loss, eva_accuracy=model.evaluate(test_dataset, verbose=1)
    # eva_loss, eva_accuracy = model.evaluate(parsed_raw_dataset_wave, parsed_raw_dataset_label,verbose=1)
    '''`y` argument is not supported when using dataset as input.'''

    #############################################################
    CM = ConfusionMatrix_label_predictor(test_dataset)
    print('################# Start Predicting:               ')


    B=tf.reduce_sum(CM,axis=1)

    print(CM)
    print(tf.reduce_sum(CM))# equals to total
    print(B)  # number of real labels

    B = tf.reshape(tf.math.reciprocal(B), [-1,1])
    # B=tf.math.reciprocal(tf.cast(B, tf.float32))
    C=tf.multiply(CM,B)
    print('sample structure:',tf.reduce_sum(C,axis=1))
    # print('transition matrix',CM)
    print('confustion matrix',C*100)



#############################################################


#########################







