import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import os
import wave
import matplotlib.pyplot as plt
import soundfile as sf


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
####################################################################
def plot_Signal(time, wave_data):
    fig = plt.figure(1,figsize=(24, 8))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.plot(time,wave_data, 'b')
    axes1.set_ylabel('Wave_Amplitude')
    axes1.set_xlabel('Time [sec]')
    axes1.set_title('Time Domain')
####################################################################



def Calculating_MFCCs_from_wave(wave_data, sample_rate):
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
    # print(spectrograms)
    # print(linear_to_mel_weight_matrix)
    linear_to_mel_weight_matrix=tf.cast(linear_to_mel_weight_matrix,dtype=tf.float64)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    print('The shape of mel_spectrograms={}'.format(mel_spectrograms.shape))
    #
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # print(spectrograms.shape[:-1])
    # print(linear_to_mel_weight_matrix.shape[-1:])
    # print('The shape of mel_spectrograms after set={}'.format(mel_spectrograms.shape))
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
    Partial = 13
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., 0:Partial]
    print('The shape of mfccs={}'.format(mfccs.shape))
    # Mean Normalization:
    # print('The shape of mfccs={} after normalization'.format(np.mean(mfccs).shape))
    mfccs -= (tf.math.reduce_mean(mfccs, axis=1))

    return mfccs

####################################################################

def read_wave_data(file_path):
    wave_data, _ = sf.read(file_path)
    with wave.open(file_path, "rb") as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # wave_data, _ = wav.read(file_path) #Unsupported bit depth: the wav file has 24-bit data.

    # 通过取样点数和取样频率计算出每个取样的时间。
    time = np.arange(0, nframes) * (1.0 / framerate)
    return wave_data, time, nchannels, sampwidth, framerate, nframes
    # return wave_data, framerate
####################################################################



####################################################################
####################################################################
####################################################################
if __name__ == "__main__":
    data_dir = r'/home/brutal/PycharmProjects/Project-tf2.3/ICBHI_final_database'


    # data_name = '184_1b1_Ar_sc_Meditron.wav' ##health all zero
    # data_name = '225_1b1_Pl_sc_Meditron.wav' ##health all zero
    data_name = '183_1b1_Pl_sc_Meditron.wav' ##health some what 1
    # data_name = '183_1b1_Tc_sc_Meditron.wav'   ##health all zero
    # data_name = '135_2b1_Pl_mc_LittC2SE.wav'  ##unhealth many 1
    # data_name = '135_2b2_Pl_mc_LittC2SE.wav'  ##unhealth all zero
    # data_name = '226_1b1_Pl_sc_LittC2SE.wav'  ##unhealth mediocre 1

    filenames = os.path.join(data_dir, data_name)


    wave_data, time, nchannels, sampwidth, sample_rate, num_samples = read_wave_data(filenames)
    wave_data_batch = np.expand_dims(wave_data, axis=0)
    print('The Number of channel={}'.format(nchannels))
    print('The Number of Sample={}'.format(num_samples))
    print('The Sample_rate={}'.format(sample_rate))
    print('The Sampwidth={}'.format(sampwidth))
    print('The shape of wave_data_batch={} and it dtype is {}\n The length of Time={} \n the time ended at {}s'
          .format(wave_data_batch.shape, type(wave_data_batch), time.shape, np.max(time)))
    # signal = wave_data[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    # plot_Signal(time[0:int(3.5 * sample_rate)], signal)
    plot_Signal(time, wave_data)
    plt.show()

    batch_size=wave_data_batch.shape[0]
    mfccs = Calculating_MFCCs_from_wave(wave_data_batch[0:,0:int(3.5 * sample_rate)], sample_rate)
    mfccs = Calculating_MFCCs_from_wave(wave_data_batch, sample_rate)

    # num_samples=int(.5 * sample_rate)
    # plot_MFCCs(mfccs, 0, num_samples, sample_rate)
    # plt.show()
    # plot_Signal(np.linspace(0,num_samples/sample_rate,mfccs.shape[1]), mfccs[0,0:,1])
    # plt.show()