import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import os
import wave
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf


####################################################################
#打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
# def read_wave_data(file_path):
#     # f = wave.open(file_path, "rb")
#     wave_data,framerate1=sf.read(file_path)
#     with wave.open(file_path, "rb") as f:
#         # 读取格式信息
#         # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采样频率, 采样点数, 压缩类型, 压缩类型的描述。
#         # wave模块只支持非压缩的数据，因此可以忽略最后两个信息
#         params = f.getparams()
#         nchannels, sampwidth, framerate, nframes = params[:4]
#         # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
#         # str_data = f.readframes(nframes)
#         # Number_of_wave_data=len(str_data)//sampwidth
#         # # 将波形数据转换成数组
#         # # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
#         # wave_data =  struct.unpack('{}b'.format(Number_of_wave_data), str_data)
#
#     # wave_data = np.frombuffer(str_data, dtype=np.int32)
#     # # 将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
#     # wave_data.shape = -1, 2
#     # wave_data = wave_data.T
#     # 通过取样点数和取样频率计算出每个取样的时间。
#     time = np.arange(0, nframes) * (1.0 / framerate)
#     return wave_data, time, nchannels, sampwidth, framerate, nframes
    # return wave_data, framerate
####################################################################
#打开wav文件 ，

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
def plot_0(time, wave_data):
	fig = plt.figure(1,figsize=(24, 8))

	axes1 = fig.add_subplot(1, 1, 1)
	axes1.plot(time,wave_data, 'b-')
	# axes1.bar(xdata,data1, color='b',width=bar_width)
	# miloc_x1 = plt.MultipleLocator(1) #
	# miloc_y1 = plt.MultipleLocator(50) #
	# axes1.xaxis.set_major_locator(miloc_x1)
	# axes1.yaxis.set_major_locator(miloc_y1)
	# axes1.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)
####################################################################

if __name__ == "__main__":
    data_dir = r'/home/brutal/PycharmProjects/Project-tf2.3/ICBHI_final_database'
    # data_name = '184_1b1_Ar_sc_Meditron.wav' ##health all zero
    # data_name = '225_1b1_Pl_sc_Meditron.wav' ##health all zero
    # data_name = '183_1b1_Pl_sc_Meditron.wav' ##health some what 1
    # data_name = '183_1b1_Tc_sc_Meditron.wav'   ##health all zero
    # data_name = '135_2b1_Pl_mc_LittC2SE.wav'  ##unhealth many 1
    data_name = '135_2b2_Pl_mc_LittC2SE.wav'  ##unhealth all zero
    # data_name = '226_1b1_Pl_sc_LittC2SE.wav'  ##unhealth mediocre 1


    filenames = os.path.join(data_dir, data_name)
    # filenames = r"/home/brutal/PycharmProjects/Project-tf2.3/HearSoundDataBase/b/Bunlabelledtest__101_1305030823364_A.wav"
    wave_data, time, nchannels, sampwidth, sample_rate, nframes = read_wave_data(filenames)
    # wave_data, framerate = read_wave_data(filenames)
    print('The Number of channel={}'.format(nchannels))
    print('The Number of Sample={}'.format(nframes))
    print('The Sample_rate={}'.format(sample_rate))
    print('The Sampwidth={}'.format(sampwidth))
    print('The length of Wave={}\n The length of Time={} and the time duriation={}'.format(len(wave_data), time.shape, np.max(time)))
    # print('{} {}'.format(len(wave_data),framerate))
    # plot_0(time, wave_data)
    # plt.show()


    # signal = wave_data[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    # plot_0(time[0:int(3.5 * sample_rate)], signal)
    signal = wave_data  # Keep the first 3.5 seconds
    plot_0(time, signal)

    plt.show()

################# Pre-Emphasis:
    # #The first step is to apply a pre-emphasis filter on the signal to amplify the high frequencies:
    # signal = wave_data
    pre_emphasis=0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
################# Framing
    #After pre-emphasis, we need to split the signal into short-time frames.
    frame_size_second= 25 / 1000 #Popular settings are 25 ms for the frame size
    frame_stride_second = 10 / 1000 # 10 ms stride
    frame_length, frame_step = frame_size_second * sample_rate, frame_stride_second * sample_rate  # Convert from seconds to samples
    print('The frame length={}'.format(frame_length))
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length ## Make sure each frame has same length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal,z)
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    #np.tile(a,(2,1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数。本例中X轴扩大一倍便为不复制。
    indices =np.tile(np.arange(0, frame_length), (num_frames, 1)) \
             + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    #改变np.array中所有数据元素的数据类型
################# Hamming Window
    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
################# Fourier-Transform and Power Spectrum
    NFFT=1024
    mag_frames = np.abs(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    print('The shape of Mpow_frames={}'.format(pow_frames.shape))
    print(np.min(mag_frames),np.max(mag_frames))

    # plot_0(np.arange(0,NFFT//2+1), mag_frames[2][:])
    # plt.show()

    #########
    t1=np.linspace(0,np.max(time+1e-3),num_frames)
    f1=np.linspace(0,sample_rate,NFFT//2+1)
    # x axis = time, y axis = frequency
    plt.pcolormesh(t1, f1, mag_frames.T, cmap=plt.cm.hot, vmin=0)
    # plt.plot(np.arange(max_f.shape[0]), max_f)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    #########

##################  Filter Banks: The final step to computing filter banks is applying triangular filters
    nfilt = 40 #number of filter
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)# Corresponding to the point in
    # num_spectrogram_bins linearly sampled frequency information from [0, sample_rate] (NFFT+1 bins)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
################## Mel-frequency Cepstral Coefficients (MFCCs)
    #################  Mean  Normalization
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    print('The shape of Mel-frequency Cepstral Coefficients={}'.format(mfcc.shape))

    t1=np.linspace(0,np.max(time+1e-3),num_frames)
    f1=np.linspace(0,12,num_ceps)
    # x axis = time, y axis = frequency
    plt.pcolormesh(t1, f1, mfcc.T, cmap=plt.cm.hot, vmin=0)
    plt.colorbar()
    # plt.plot(np.arange(max_f.shape[0]), max_f)
    plt.ylabel('MFCCs')
    plt.xlabel('Time [sec]')
    plt.show()


















