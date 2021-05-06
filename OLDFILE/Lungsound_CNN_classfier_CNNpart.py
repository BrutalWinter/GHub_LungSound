import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models # FOR SPEEED UP PURPOSE
import matplotlib.pyplot as plt
###################################


def plot_history_all(history):
    history_dict = history.history
    print(history_dict.keys())  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']


    epochs = range(1, len(acc) + 1)
    fig = plt.figure(2,figsize=(15, 10),dpi=200) #开启一个窗口，同时设置大小，分辨率
    axes1 = fig.add_subplot(2, 1, 1) #通过fig添加子图，参数：行数，列数，第几个。
    axes2 = fig.add_subplot(2, 1, 2)

    # “bo”代表 "蓝点"    # b代表“蓝色实线”
    axes1.plot(epochs, loss, 'bo', label='Training loss')
    axes1.plot(epochs, val_loss, 'b', label='Validation loss')
    axes2.plot(epochs, acc, 'ro', label='Training accuracy')
    axes2.plot(epochs, val_acc, 'r', label='Validation accuracy')

    axes1.set_ylabel('Loss')
    axes1.set_xlabel('Epochs')
    axes1.set_title('Training and validation loss')
    axes1.legend(loc='upper left')

    axes2.set_ylabel('accuracy')
    axes2.set_xlabel('Epochs')
    axes2.set_title('Training and validation accuracy')
    axes2.legend(loc='upper left')

def plot_MFCCs(mfccs,index,num_samples, sample_rate):
    num_frames_per_sample=mfccs.shape[1]
    num_mfcc_per_frame= mfccs.shape[-1]

    time_end=num_samples/sample_rate
    time_step=time_end/(num_frames_per_sample-1)
    time_x_axis, y_axis = np.mgrid[0:(time_end+0.5*time_step):time_step, 0:num_mfcc_per_frame:1]
    # time_x_axis=np.linspace(0,time_end,num_frames_per_sample+1)
    # y_axis=np.arange(0,num_mfcc_per_frame)
    mfccs_data=mfccs[index]

    fig = plt.figure(2,figsize=(18, 5),dpi=200) #开启一个窗口，同时设置大小，分辨率
    axes1 = fig.add_subplot(1, 1, 1) #通过fig添加子图，参数：行数，列数，第几个。
    im1=axes1.pcolormesh(time_x_axis, y_axis, mfccs_data, shading='auto', cmap=plt.cm.cool)
    fig.colorbar(im1, ax=axes1)
    # plt.plot(np.arange(max_f.shape[0]), max_f))
    axes1.set_ylabel('MFCCs')
    axes1.set_xlabel('Time [sec]')
    axes1.set_title('MFCCs heatmap')
    miloc_y1 = plt.MultipleLocator(1) #
    miloc_x1 = plt.MultipleLocator(0.5)  #
    axes1.yaxis.set_major_locator(miloc_y1)
    axes1.xaxis.set_major_locator(miloc_x1)

def plot_Signal(time, wave_data):
    fig = plt.figure(1,figsize=(24, 8))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.plot(time,wave_data, 'b')
    axes1.set_ylabel('Wave_Amplitude')
    axes1.set_xlabel('Time [sec]')
    axes1.set_title('Time Domain')

def Calculating_MFCCs_from_wave(wave_data, sample_rate):
    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    # wave_data=tf.cast(wave_data, dtype=tf.float32)
    frame_length,frame_step,fft_length = 1024, 256, 1024
    stfts = tf.signal.stft(wave_data, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, window_fn=tf.signal.hamming_window,
                           pad_end=True)
    spectrograms = tf.math.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    print('The shape of STFTs={}'.format(spectrograms.shape))
    num_spectrogram_bins = stfts.shape[-1]
    # print(num_spectrogram_bins) # where fft_unique_bins is fft_length // 2 + 1 (the unique components of the FFT).

    nfilt = 40
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, sample_rate / 2, nfilt
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate,
                                                                        lower_edge_hertz, upper_edge_hertz)
    print('filter bank shape={}'.format(linear_to_mel_weight_matrix.shape))
    ########## show the filter banks:##########
    # plot_FilterBanks(linear_to_mel_weight_matrix)
    # plt.show()
    #################### ##########
    linear_to_mel_weight_matrix=tf.cast(linear_to_mel_weight_matrix,dtype=tf.float32)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    print('The shape of mel_spectrograms={}'.format(mel_spectrograms.shape))

    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # print(spectrograms.shape[:-1])
    # print(linear_to_mel_weight_matrix.shape[-1:])
    # print('The shape of mel_spectrograms after set={}'.format(mel_spectrograms.shape))

    # Compute a stabilized mel-scale spectrograms.
    mel_spectrograms = tf.where(mel_spectrograms <= 0, np.finfo(float).eps, mel_spectrograms)
    # A=np.array([[1,-1],[-2.,3],[4,5]])
    # print(A)
    # A=tf.where(A<=0,np.finfo(float).eps,A)
    # print(A)
    # log to get log-magnitude
    log_mel_spectrograms = tf.math.log(mel_spectrograms)  # .e-base
    # Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.
    print('The shape of log_mel_spectrograms={}'.format(log_mel_spectrograms.shape))
    #
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    Partial = 14
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., 0:Partial]
    # Mean Normalization:
    # mean_mfccs=tf.math.reduce_mean(mfccs, axis=1)
    # mean_mfccs = np.expand_dims(mean_mfccs, axis=1)
    # mean_mfccs_tiles=tf.tile(mean_mfccs,[1,mfccs.shape[1],1])
    # mfccs -= mean_mfccs_tiles

    print('The shape of mfccs={}'.format(mfccs.shape))


    return mfccs
################################### data test using mnist:
# mnist = tf.keras.datasets.mnist
# (train, train_label), (test, test_label) = mnist.load_data()
# train, test = train / 255.0, test / 255.0
# # Add a channels dimension
# train = train[..., tf.newaxis]
# test = test[..., tf.newaxis]
# print('The shape of train_images={}\n The length of train_labels={}'.format(train.shape, len(train_label)))
# print('label example={}'.format(train_label[0:12]))
#########################################################################################################
if __name__ == "__main__":
    batch_size, num_samples, sample_rate = 500, 32000, 8000.0
    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    pcm = tf.random.normal([batch_size, num_samples], dtype=tf.float32)

    mfccs = Calculating_MFCCs_from_wave(pcm, sample_rate)
    plot_MFCCs(mfccs, 0, num_samples, sample_rate)
    plt.show()


    mfccs= mfccs[..., tf.newaxis]
    label=np.random.randint(10, size=(batch_size,1))
    print(label.shape)


    input_shape = mfccs.shape[-3:]
    print('Input Shape={}'.format(input_shape))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 1), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 1), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1), padding='valid'),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1), padding='valid'),

        # tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        # tf.keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu'),
        # tf.keras.layers.MaxPooling2D((2, 2),strides=(2, 2),padding='valid'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),  # Dense 层的输入为向量（一维）
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),  # Dense 层的输入为向量（一维）
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, kernel_regularizer='l2', kernel_initializer=tf.keras.initializers.GlorotUniform())
    ])
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    #
    # ################# cross validation
    val_data = mfccs[:100]
    train = mfccs[100:]
    val_label = label[:100]
    train_label = label[100:]

    test= mfccs[120:250]
    test_label= label[120:250]
    #
    history = model.fit(train, train_label, batch_size=50, epochs=6, validation_data=(val_data, val_label))
    test_loss, test_acc = model.evaluate(test, test_label, verbose=1)
    #
    # plot_history_all(history)
    # plt.show()

















