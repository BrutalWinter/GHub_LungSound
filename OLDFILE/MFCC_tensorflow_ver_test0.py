import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import os
import wave
import matplotlib.pyplot as plt
import soundfile as sf
'''语音是时间序列，其归一化的均值和方差并非是一个矩阵而是一维的向量。
100条语句，假设总共有1000帧数据，假设每帧数据为40纬的FBANK特征，那么归一化的两个参数:均值和标准差也会是纬度为40的向量。
归一化的均值为1000帧40纬特征的均值，标准差为这1000帧特征的标准差。归一化的过程就是每个特征减去均值，并除以标准差

可以把一条语音看成长度可变，宽度(mfcc,fbank等)固定的图像。
假如宽度为40维的mfcc,那么归一化有两种方式:1、在整个语料中统计所有的这40维的均值方差，最后得到的是2个全局的40维向量；
2、每句话统计一下2个40维的均值方差，n句话就有2n个向量。  两种方式都和语音长度无关。
。'''



####################################################################
def plot_FilterBanks(Filterbanks):
    # fig = plt.figure(1, figsize=(24, 8))
    # axes1 = fig.add_subplot(1, 1, 1)
    plt.figure(1, figsize=(24, 8))
    x_data = np.arange(0, Filterbanks.shape[0])
    for i in np.arange(0, Filterbanks.shape[-1]):
        plt.plot(x_data, Filterbanks[0:,i], 'k')

    # axes1.bar(xdata,data1, color='b',width=bar_width)
    # miloc_x1 = plt.MultipleLocator(1) #
    # miloc_y1 = plt.MultipleLocator(50) #
    # axes1.xaxis.set_major_locator(miloc_x1)
    # axes1.yaxis.set_major_locator(miloc_y1)
    # axes1.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)

####################################################################
def plot_MFCCs(mfccs,index,num_samples, sample_rate):
    num_frames_per_sample=mfccs.shape[1]
    num_mfcc_per_frame= mfccs.shape[-1]

    time_end=num_samples/sample_rate
    time_step=time_end/(num_frames_per_sample-1)
    time_x_axis, y_axis = np.mgrid[0:(time_end+0.5*time_step):time_step, 0:num_mfcc_per_frame:1]
    mfccs_data=mfccs[index]

    fig = plt.figure(2,figsize=(10, 10),dpi=240) #开启一个窗口，同时设置大小，分辨率
    axes1 = fig.add_subplot(1, 1, 1) #通过fig添加子图，参数：行数，列数，第几个。
    im1=axes1.pcolormesh(time_x_axis, y_axis, mfccs_data, shading='auto', cmap=plt.cm.rainbow)
    fig.colorbar(im1, ax=axes1)
    # plt.plot(np.arange(max_f.shape[0]), max_f))
    axes1.set_ylabel('MFCCs')
    axes1.set_xlabel('Time [sec]')
    axes1.set_title('MFCCs heatmap')
####################################################################


def Calculating_MFCCs_from_wave(wave_data, batch_size, num_samples, sample_rate):
    # A 1024-point STFT with frames of 64 ms and 75% overlap.

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
    # A=linear_to_mel_weight_matrix[0:,1]
    # print(A)
    ########## show the filter banks:##########
    # plot_FilterBanks(linear_to_mel_weight_matrix)
    # plt.show()
    #################### ##########
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    print('The shape of mel_spectrograms={}'.format(mel_spectrograms.shape))
    #
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    print(spectrograms.shape[:-1])
    print(linear_to_mel_weight_matrix.shape[-1:])
    print('The shape of mel_spectrograms after set={}'.format(mel_spectrograms.shape))
    #
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
    Partial = 15
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., 0:Partial]
    print('The shape of mfccs={}'.format(mfccs.shape))
    plot_MFCCs(mfccs, np.random.randint(0, batch_size), num_samples, sample_rate)
    plt.show()










####################################################################
batch_size, num_samples, sample_rate = 32, 32000, 16000.0
# A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
pcm = tf.random.normal([batch_size, num_samples], dtype=tf.float32)
plot_FilterBanks(tf.transpose(pcm[0:1,0:]))
plt.show()
Calculating_MFCCs_from_wave(pcm,batch_size, num_samples, sample_rate)
####################################################################
