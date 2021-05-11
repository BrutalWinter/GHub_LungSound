import os
import tensorflow as tf
import numpy as np
import soundfile as sf #pysoundfile 全部自动转换成float32来输出 numpy result
import matplotlib.pyplot as plt
########################################################
def plt_wav_batch(wave_data, id, Label_batch, sample_rate):
    # wave_data: [N, Sample_len]
    Num_per_batch = wave_data.shape[0]
    Sample_len = wave_data.shape[1]
    time_axis = np.arange(0, Sample_len) * (1.0 / sample_rate)

    fig = plt.figure(figsize=(16, 8))
    for ith, wave in enumerate(wave_data):
        ax1 = fig.add_subplot(Num_per_batch, 1, ith+1)
        ax1.plot(time_axis,wave, color='b')

        ax1.set_title("wav data {} and label={}".format(id[ith],Label_batch[ith]))
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("Amplitude")
    plt.show()



def plt_spectrum_from_wave_batchs(wave_data, id, Label_batch, sample_rate, FFT_len, overlap):
    # wave_data: [N, Sample_len]  it stft pad_end=False
    Num_per_batch = wave_data.shape[0]
    Sample_len = wave_data.shape[1]
    time = Sample_len / sample_rate

    fig = plt.figure(figsize=(18, 8))
    for ith, wave in enumerate(wave_data):
        ax1 = fig.add_subplot(Num_per_batch, 1, ith + 1)
        spectrum, freqs, ts, im1 = ax1.specgram(wave,NFFT=FFT_len, Fs=sample_rate, noverlap=overlap, window=np.hamming(FFT_len), xextent=(0, time), cmap='viridis')

        ax1.set_title("wav spectrum {} and label={} for compare".format(id[ith],Label_batch[ith]))
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("Frequency(Hz)")
        plt.colorbar(im1)

        print('freqs.shape',freqs.shape)
        print('ts.shape', ts.shape)
    plt.show()



def plot_FilterBanks(Filterbanks):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    x_data = np.arange(0, Filterbanks.shape[0])
    ax1.plot(x_data, Filterbanks[0:, 0], 'k')
    # for i in np.arange(0, Filterbanks.shape[-1]):
    #     ax1.plot(x_data, Filterbanks[0:,i], 'k')
    plt.show()



def plt_MFCC_batch(mfccs_batch, id, Label_batch, time_duration):
    # mfccs_batch: [N, n_frames, n_mfccs_coes]
    Num_per_batch = mfccs_batch.shape[0]
    Num_frames_per_sample = mfccs_batch.shape[1]
    Num_mfcc_per_frame = mfccs_batch.shape[2]

    # time_axis, mfcc_axis= np.mgrid[0:(time_end+0.5*time_step):time_step, 0:num_mfcc_per_frame:1]
    time_axis=np.linspace(0, time_duration,num=Num_frames_per_sample+1)
    mfcc_axis=np.arange(0, Num_mfcc_per_frame+1)

    fig = plt.figure(figsize=(18, 8))
    for ith, mfcc in enumerate(mfccs_batch):
        # print(mfcc.shape)
        mfcc=tf.transpose(mfcc)
        ax1 = fig.add_subplot(Num_per_batch, 1, ith + 1)
        im1 = ax1.pcolormesh(time_axis, mfcc_axis, mfcc, shading='auto', cmap='viridis')


        ax1.set_title("MFCCs heatmap {} and label={}".format(id[ith],Label_batch[ith]))
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("MFCC-Coefficients")

        # miloc_y1 = plt.MultipleLocator(4)
        # miloc_x1 = plt.MultipleLocator(1)
        # ax1.yaxis.set_major_locator(miloc_y1)
        # ax1.xaxis.set_major_locator(miloc_x1)
        plt.colorbar(im1)
    plt.show()
    # save_path = r'/home/brutal/PycharmProjects/Project-lungsound/FIGs'
    # save_name = '{}.png'.format(id)
    # save_file_path = os.path.join(save_path, save_name)
    # plt.savefig(save_file_path)

def plt_spectrogram_batch(spectrogram_batch, id, Label_batch, time_duration, sample_rate=8000):
    Num_per_batch = spectrogram_batch.shape[0]
    Num_frames_per_sample = spectrogram_batch.shape[1]
    Num_freqs_per_frame = spectrogram_batch.shape[2]

    time_axis=np.linspace(0, time_duration, num=Num_frames_per_sample)
    freqs_axis=np.linspace(0, sample_rate//2, num=Num_freqs_per_frame)

    fig = plt.figure(figsize=(18, 8))
    for ith, freqs in enumerate(spectrogram_batch):
        # Convert to frequencies to log scale and transpose so that the time is represented in the x-axis (columns).
        freqs = tf.math.log(tf.transpose(freqs))
        ax1 = fig.add_subplot(Num_per_batch, 1, ith + 1)
        im1 = ax1.pcolormesh(time_axis, freqs_axis, freqs, shading='gouraud', cmap='viridis')

        ax1.set_title("wav spectrogram {} and label={}".format(id[ith],Label_batch[ith]))
        ax1.set_xlabel("Time(s)")
        ax1.set_ylabel("Frequency(Hz)")

        plt.colorbar(im1)
    plt.show()

    # save_path = r'/home/brutal/PycharmProjects/Project-lungsound/FIGs'
    # save_name = '{}.png'.format(id)
    # save_file_path = os.path.join(save_path, save_name)
    # plt.savefig(save_file_path)







###########################################################################
def wave_resampled(wave_data, Fs_original, Fs_desire):
    time_total = len(wave_data) // Fs_original
    x_before= np.linspace(0, time_total, len(wave_data))
    x_after = np.linspace(0, time_total, time_total * Fs_desire)
    wave_data_resampled = np.interp(x_after, x_before, wave_data)
    return wave_data_resampled


#梅尔倒谱系数（mfcc）, 连续语音提取过程： 预加重--加窗分帧--FFT--abs--MEL滤波器组--对数运算--DCT
def Pre_Emphasis_data(wave_data_batch,emphasis_coefficent=0.97):
    Emphasis_wave_data_batch=wave_data_batch[...,1:]-wave_data_batch[...,:-1]*emphasis_coefficent
    Emphasis_wave_data_batch=tf.concat([wave_data_batch[...,0:1],Emphasis_wave_data_batch],axis=-1)
    return Emphasis_wave_data_batch


# 为什么汉明窗这样取呢？因为之后我们会对汉明窗中的数据进行FFT，它假设一个窗内的信号是代表一个周期的信号。 典型的窗口大小是25ms，帧移是10ms (15ms overlap)
# （也就是说窗的左端和右端应该大致能连在一起）而通常一小段音频数据没有明显的周期性，加上汉明窗后，数据形状就有点周期的感觉了。
# 因为加上汉明窗，只有中间的数据体现出来了，两边的数据信息丢失了，所以等会移窗的时候，只会移1/3或1/2窗(overlap)，这样被前一帧或二帧丢失的数据又重新得到了体现。
def Calculating_MFCCs_from_wave(PCM_data_batch, sample_rate,
                                window_frame_len=1024, frame_step=256,fft_length=1024, num_mel_bins = 80, num_mel_keepbins=20):
    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    ################## if needed:
    # zero_padding = tf.zeros([8000*20] - tf.shape(waveform), dtype=tf.float32)# Padding for files
    # # Concatenate audio with padding so that all audio clips will be of the same length
    # waveform = tf.cast(waveform, tf.float32)
    # PCM_data_batch = tf.concat([waveform, zero_padding], 0)
    ##################

    stfts = tf.signal.stft(PCM_data_batch,
                           frame_length=window_frame_len, frame_step=frame_step, fft_length=fft_length, window_fn=tf.signal.hamming_window, pad_end=False)
    spectrograms = tf.math.abs(stfts)  #[N, frames, fft_unique_bins=fft_length//2 + 1]
    # Warp the linear scale spectrograms into the mel-scale:
    num_spectrogram_bins = stfts.shape[-1] # num of frames

    low_freq_mel = 0
    # F_limit = 0.5 * sample_rate
    F_limit=1000
    racf=tf.math.log(1+(F_limit)/700) / tf.math.log(10.0)
    high_freq_mel = 2595 * racf
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                sample_rate,lower_edge_hertz=low_freq_mel, upper_edge_hertz=high_freq_mel) #A Tensor of shape [num_spectrogram_bins, num_mel_bins].
    ###########################################
    # plot_FilterBanks(linear_to_mel_weight_matrix)
    ###########################################
    # linear_to_mel_weight_matrix=tf.cast(linear_to_mel_weight_matrix,dtype=tf.float32)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)

    # With eager execution this operates as a shape assertion, the shapes match:
    #tf.TensorShape.concatenate: Returns the concatenation of the dimension in self and other.
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)# Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)# Compute MFCCs from log_mel_spectrograms.
    mfccs = mfccs[..., :num_mel_keepbins]#and take the first 20 because 倒谱系数后，我们只需要取低位的系数便可以得到包络信息


    # print('low_freq_mel',low_freq_mel)
    # print('high_freq_mel', high_freq_mel)
    # print('The shape of spectrogram (STFTs)={}'.format(spectrograms.shape),spectrograms.dtype)
    # print('num_spectrogram_bins=',num_spectrogram_bins)
    # print('Mel-filter bank={}'.format(linear_to_mel_weight_matrix.shape), linear_to_mel_weight_matrix.dtype)
    # print('The shape of mel_spectrograms={}'.format(mel_spectrograms.shape),mel_spectrograms.dtype)
    # print('mel_spectrograms.set_shape', spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # print('The shape of mfccs={}'.format(mfccs.shape),mfccs.dtype)
    # print('\n')
    return spectrograms, mfccs













##########################################
##########################################
if __name__ == "__main__":
    ############################### single:
    path=r'/home/brutal/PycharmProjects/Data_base/dataset-lunsound/ICBHI_final_database/101_1b1_Al_sc_Meditron.wav'
    sample_data, sample_rate = sf.read(path)
    print(sample_data)
    print(sample_rate)
    print(len(sample_data))
    fig1 = plt.figure(figsize=(25, 10))
    axes1 = fig1.add_subplot(1, 1, 1)
    axes1.plot(sample_data)
    plt.show()

    ################################ alll:
    Audio_Directory = r'/home/brutal/PycharmProjects/Data_base/dataset-lunsound/ICBHI_final_database'
    Audio_files_path = os.path.join(Audio_Directory, '*.wav')
    Audio_files = sorted(tf.io.gfile.glob(Audio_files_path))
    # print(Audio_files)
    print('Audio_files:')
    for i in Audio_files:
        # print(i)
        sample_datas, sample_rates = sf.read(i)
        print(sample_datas.shape)
        print(sample_rates)

        fig1 = plt.figure(figsize=(25, 10))
        axes1 = fig1.add_subplot(1, 1, 1)
        axes1.plot(sample_datas)
        plt.show()




