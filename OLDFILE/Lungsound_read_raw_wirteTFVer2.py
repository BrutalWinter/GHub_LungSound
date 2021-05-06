import tensorflow as tf
import numpy as np
import pandas as pd

##########tf.io.gfile.glob equals glob
import struct
import os
import matplotlib.pyplot as plt
import wave
import csv
import sys
import soundfile as sf
import glob
####################################################################################
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value)) # becareful!!!!!!!!!!!!!!about[A] and A

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  #'''list of floats can not be list of list'''

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_data(efid_index_cycle, wave_data, label):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'efid_index_cycle': _int64_feature(efid_index_cycle),
      'label': _int64_feature(label),
      'wave_data': _float_feature(wave_data),
      # 'wave_data': _bytes_feature(wave_data),

  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
####################################################################################
####################################################################################



# def WaveData_to_tfrecords(record_file):
#     for i in range(n_observations):
#         example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
#         writer.write(example)

def plot_sound_wave(wave_data,framerate):
    time_x_axis=np.linspace(0,len(wave_data)/framerate,len(wave_data))
    fig = plt.figure(2,figsize=(15, 5),dpi=120) #开启一个窗口，同时设置大小，分辨率
    axes1 = fig.add_subplot(1, 1, 1) #通过fig添加子图，参数：行数，列数，第几个。
    axes1.plot(time_x_axis,wave_data, 'r')
    # plt.plot(np.arange(max_f.shape[0]), max_f))
    axes1.set_ylabel('SoundWave Amplitude')
    axes1.set_xlabel('Time [sec]')
    axes1.set_title('SoundWave Figure')
    plt.show()

def wave_resampled(wave_data, Fs_pre, Fs_after):
    time_total=len(wave_data)/Fs_pre
    number_of_points=time_total*Fs_after
    x_before=np.linspace(0,time_total,len(wave_data))
    x_after = np.linspace(0, time_total,np.ceil(number_of_points).astype(np.int))
    wave_data_resampled = np.interp(x_after, x_before, wave_data)
    return wave_data_resampled


def Calculating_MFCCs_from_wave(wave_data, sample_rate):
    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    # print('length of data={}'.format(len(wave_data)))
    # print('shape of data={}\n\n\n\n'.format(wave_data.shape))
    # if len(wave_data)==0:# 108th data has problem!
    #     wave_data = tf.random.normal([1024], dtype=tf.float32)
    # else:
    #     pass


    frame_length,frame_step,fft_length = 512, 128, 512
    stfts = tf.signal.stft(wave_data, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, window_fn=tf.signal.hamming_window,
                           pad_end=True)
    spectrograms = tf.math.abs(stfts)
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    # print(num_spectrogram_bins) # where fft_unique_bins is fft_length // 2 + 1 (the unique components of the FFT).
    nfilt = 50
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, sample_rate / 2, nfilt
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate,
                                                                        lower_edge_hertz, upper_edge_hertz)
    # print('The shape of STFTs={}'.format(spectrograms.shape))
    # print('filter bank shape={}'.format(linear_to_mel_weight_matrix.shape))
    ########## show the filter banks:##########
    # plot_FilterBanks(linear_to_mel_weight_matrix)
    # plt.show()
    #################### ##########
    linear_to_mel_weight_matrix = tf.cast(linear_to_mel_weight_matrix, dtype=tf.float64)
    spectrograms = tf.cast(spectrograms, dtype=tf.float64)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    # print('The shape of mel_spectrograms={}'.format(mel_spectrograms.shape))
    #print(ar3[:-1])==print(ar3[0:-1])     print(ar3[-1:])==[print(ar3[-1])]
    mel_spectrograms.set_shape(spectrograms.shape[0:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # print('The shape of mel_spectrograms={}'.format(mel_spectrograms.shape))
    # print('The shape of mel_spectrograms after set={}'.format(mel_spectrograms.shape))
    #
    # Compute a stabilized mel-scale spectrograms.
    mel_spectrograms = tf.where(mel_spectrograms <= 0, np.finfo(float).eps, mel_spectrograms)
    log_mel_spectrograms = tf.math.log(mel_spectrograms)  # .e-base
    # Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.
    # print('The shape of log_mel_spectrograms={}'.format(log_mel_spectrograms.shape))
    #
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    Partial = 24
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., 0:Partial]

    plot_MFCCs(mfccs, len(wave_data), sample_rate)
    plt.show()
    # mfccs=np.mean(mfccs,axis=0)
    # print('The shape of mfccs={} and Length of wave_data={}'.format(mfccs.shape, len(wave_data)))



    return mfccs


def process_raw_data(des_file, data_file, label_file, Fs, record_file):
    piece_len = 20
    file_len = os.path.getsize(des_file)
    Num_Wave_frames=file_len // piece_len ## 920 intotal
    print('The total number of wave data in database={}'.format(Num_Wave_frames))

    with tf.io.TFRecordWriter(record_file) as writer:
        with open(des_file, 'rb') as data_handler:
            label_dicts = {}  #:dict be careful
            wave_data_dict = {}  #:dict be careful
            wave_num_cycle_dict = {}
            wave_data_max_len = []
            Fs_after = Fs
            # The_i_th_num_of_lungsound_cycle=[0]
            The_i_th_num_of_lungsound_cycle = [4788+1]
            # for i in range(Num_Wave_frames):  # total of Num_Wave_frames, each Num_Wave_frames contains different num_of_cycle
            start_Wave_Num=np.ceil(0.7*Num_Wave_frames).astype(np.int32) #The 4788th sample label=0
            for i in range(start_Wave_Num,Num_Wave_frames):
            # for i in range(start_Wave_Num):
                data_handler.seek(piece_len * i, 0)
                if i > Num_Wave_frames:
                    break
                data = data_handler.read(piece_len)
                print('Processing the {}th data'.format(i))
                efid, = struct.unpack('<I', data[:4])
                seek_pos_wave, = struct.unpack('<I', data[4:8])
                piece_len_wave, = struct.unpack('<I', data[8:12])
                seek_pos_label, = struct.unpack('<I', data[12:16])
                piece_len_label, = struct.unpack('<I', data[16:20])

                wave_data = get_data(data_file, seek_pos_wave, piece_len_wave)

                framerate, len_label, labels, data_strat_end = view_label(label_file, seek_pos_label, piece_len_label)


                wave_after_resampled = wave_resampled(wave_data, framerate, Fs_after)



                # wave_data_np = np.asarray(wave_data)
                plot_sound_wave(wave_data, framerate)
                # plot_sound_wave(wave_after_resampled, Fs_after)
                # print('wave_data={}'.format(len(wave_after_resampled)))
                # print('framerate={}'.format(framerate))
                # print('framerate_after={}'.format(Fs_after))
                # print('len_label={}'.format(len_label))
                # print('labels={}'.format(labels))
                # print('data_strat_end={}'.format(data_strat_end))

                label_arr = np.asarray(labels).reshape(-1, 2).tolist()  # 将数组或者矩阵转换成列表
                # label_dicts[efid] = label_arr
                data_strat_end_point = np.asarray(data_strat_end).reshape(-1, 2) * Fs_after

                # wave_after_resampled=wave_data
                # data_strat_end_point = np.asarray(data_strat_end).reshape(-1, 2) * framerate

                data_mfccs_segs = []
                data_segs_len = []
                num_of_lungsound_cycle = len(data_strat_end_point)

                # print('labels after processed={}'.format(label_dicts))
                # print('data_strat_end={}'.format(data_strat_end_point))
                print('num_of_lungsound_cycle={}'.format(num_of_lungsound_cycle))
                The_i_th_num_of_lungsound_cycle.append(num_of_lungsound_cycle)
                The_i_th_num_of_lungsound_cycle1=np.sum(The_i_th_num_of_lungsound_cycle)-The_i_th_num_of_lungsound_cycle[-1]


                for i in range(num_of_lungsound_cycle):
                    start_point = np.round(data_strat_end_point[i][0]).astype(np.uint32)
                    end_point = np.round(data_strat_end_point[i][1]).astype(np.uint32)
                    data_seg = np.array(wave_after_resampled[start_point:end_point])
                    # print(data_seg)
                    # data_seg = data_seg.astype(np.string_)


                    # print('start_point:', start_point)
                    # print('end_point:', end_point)
                    # print('Length of data_seg=          {}'.format(len(data_seg)))
                    # print(len(wave_after_resampled))
                    label_value = label_arr[i][0] * 2 + label_arr[i][1]
                    '''0 is healthy;1 is Crackle;2 is Wheeze ; 3 is both'''
                    efid_index_cycle = np.asarray(The_i_th_num_of_lungsound_cycle1 + i, dtype=np.uint32)
                    # print(efid_index_cycle,label_arr[i])
                    print('The {}th sample label={}'.format(efid_index_cycle, label_value))


                    mfccs=Calculating_MFCCs_from_wave(data_seg, Fs)
                    data_seg_MFCCs = np.mean(mfccs, axis=0)
                    # if label_value==3:
                    #     print('starting plot')
                    #     plot_sound_wave(data_seg, Fs_after)
                        # plot_MFCCs(mfccs, len(wave_data), Fs)
                        # plt.show()

                    # print('data_seg:', data_seg)

                    data_mfccs_segs.append(data_seg)
                    data_segs_len.append(len(data_seg))


                    example = serialize_data([efid_index_cycle], data_seg_MFCCs, [label_value])
                    example = serialize_data([efid_index_cycle], data_seg, [label_value])
                    writer.write(example)




                # wave_data_dict[efid] = data_mfccs_segs
                # wave_num_cycle_dict[efid] = num_of_lungsound_cycle
                # wave_data_max_len.append(np.max(data_segs_len))
                #
                # for k, k2, k3 in zip(wave_data_dict, label_dicts, wave_num_cycle_dict):
                #     num = wave_num_cycle_dict[k3]
                #     for i in range(num):
                #         efid_index_cycle=np.asarray([k,i],dtype=np.uint32)
                #         # print(efid_index_cycle)
                #         # example = serialize_data(efid_index_cycle, wave_data_dict[k][i], label_dicts[k2][i])
                #         example = serialize_data(efid_index_cycle, wave_data_dict[k][i], efid_index_cycle)
                #         writer.write(example)


# def process_raw_data1(des_file, data_file, label_file, Fs, record_file):
#     piece_len = 20
#     file_len = os.path.getsize(des_file)
#     Num_Wave_frames = file_len // piece_len
#     print('The total number of wave data in database={}'.format(Num_Wave_frames))
#
#     with open(des_file, 'rb') as data_handler:
#         label_dicts = {}  #:dict be careful
#         wave_data_dict = {}  #:dict be careful
#         wave_num_cycle_dict = {}
#         wave_data_max_len = []
#         Fs_after = Fs
#         for i in range(
#                 Num_Wave_frames):  # total of Num_Wave_frames, each Num_Wave_frames contains different num_of_cycle
#             # if i > Num_Wave_frames:
#             if i > 2:
#                 break
#             data = data_handler.read(piece_len)
#             efid, = struct.unpack('<I', data[:4])
#             seek_pos_wave, = struct.unpack('<I', data[4:8])
#             piece_len_wave, = struct.unpack('<I', data[8:12])
#             seek_pos_label, = struct.unpack('<I', data[12:16])
#             piece_len_label, = struct.unpack('<I', data[16:20])
#
#             wave_data = get_data(data_file, seek_pos_wave, piece_len_wave)
#             framerate, len_label, labels, data_strat_end = view_label(label_file, seek_pos_label, piece_len_label)
#             # wave_data_np = np.asarray(wave_data)
#
#             wave_after_resampled = wave_resampled(wave_data, framerate, Fs_after)
#
#             # plot_sound_wave(wave_data, framerate)
#             # plot_sound_wave(wave_data_np, framerate)
#             # plot_sound_wave(wave_after_resampled, Fs_after)
#             # print('wave_data={}'.format(len(wave_after_resampled)))
#             # print('framerate={}'.format(framerate))
#             # print('framerate_after={}'.format(Fs_after))
#             # print('len_label={}'.format(len_label))
#             # print('labels={}'.format(labels))
#             # print('data_strat_end={}'.format(data_strat_end))
#
#             # data_preprocess(efid, wave_data, framerate, labels, data_strat_end, label_pairs, wave_data_pairs)
#             label_arr = np.asarray(labels).reshape(-1, 2).tolist()  # 将数组或者矩阵转换成列表
#             label_dicts[efid] = label_arr
#             data_strat_end_point = np.asarray(data_strat_end).reshape(-1, 2) * Fs_after
#
#             data_segs = []
#             data_segs_len = []
#             num_of_lungsound_cycle = len(data_strat_end_point)
#
#             # print('labels after processed={}'.format(label_dicts))
#             # print('data_strat_end={}'.format(data_strat_end_point))
#             # print('num_of_lungsound_cycle={}'.format(num_of_lungsound_cycle))
#
#             for i in range(num_of_lungsound_cycle):
#                 start_point = np.floor(data_strat_end_point[i][0]).astype(np.uint32)
#                 end_point = np.ceil(data_strat_end_point[i][1]).astype(np.uint32)
#                 data_seg = np.array(wave_after_resampled[start_point:end_point])
#
#                 # print('start_point:', start_point)
#                 # print('end_point:', end_point)
#                 # print('Length of data_seg=          {}'.format(len(data_seg)))
#
#                 # print('data_seg:', data_seg)
#                 data_segs.append(data_seg)
#                 data_segs_len.append(len(data_seg))
#             wave_data_dict[efid] = data_segs
#             wave_num_cycle_dict[efid] = num_of_lungsound_cycle
#             wave_data_max_len.append(np.max(data_segs_len))
#             # print('wave_data processed={}'.format(wave_data_pairs))
#             # print('wave_data_max_len={}'.format(wave_data_max_len))
#     return wave_data_dict, label_dicts, wave_num_cycle_dict, wave_data_max_len


def get_data(data_file, seek_pos_wave, piece_len_wave):
    with open(data_file, 'rb') as data_handler:
        data_handler.seek(seek_pos_wave)
        data = data_handler.read(piece_len_wave)
        efid_wave, = struct.unpack('<I', data[:4])

        len_wave_data, = struct.unpack('<I', data[4:8])
        wave_data = struct.unpack('<' + 'f' * len_wave_data, data[8:])
    return wave_data


def view_label(label_file, seek_pos_label, piece_len_label):
    with open(label_file, 'rb') as data_handler:
        data_handler.seek(seek_pos_label)
        data = data_handler.read(piece_len_label)
        efid_label, = struct.unpack('<I', data[:4])
        framerate, = struct.unpack('<I', data[4:8])
        len_label, = struct.unpack('<I', data[8:12])
        # print('len_label', len_label)
        labels = struct.unpack('<' + 'I' * len_label, data[12:12 + len_label * 4])
        data_strat_end = struct.unpack('<' + 'f' * len_label, data[12 + len_label * 4:])


    return framerate, len_label, labels, data_strat_end



def plot_MFCCs(mfccs,num_samples, sample_rate):
    num_frames_per_sample=mfccs.shape[0]
    num_mfcc_per_frame= mfccs.shape[1]
    time_x_axis, y_axis = np.mgrid[0:num_frames_per_sample+1, 0:num_mfcc_per_frame+1]


    fig = plt.figure(2,figsize=(12, 6),dpi=120) #开启一个窗口，同时设置大小，分辨率
    axes1 = fig.add_subplot(1, 1, 1) #通过fig添加子图，参数：行数，列数，第几个。
    im1=axes1.pcolormesh(time_x_axis, y_axis, mfccs, shading='auto', cmap=plt.cm.rainbow)
    fig.colorbar(im1, ax=axes1)
    # plt.plot(np.arange(max_f.shape[0]), max_f))

    time_end = num_samples / sample_rate



    x_label = np.linspace(0, time_end, num_frames_per_sample + 1)
    # print(x_label)

    xmajorLocator = plt.MultipleLocator(5)  # 定义横向主刻度标签的刻度差为2的倍数。就是隔几个刻度才显示一个标签文本
    axes1.xaxis.set_major_locator(xmajorLocator)  # x轴 应用定义的横向主刻度格式。如果不应用将采用默认刻度格式
    A=(1 + np.floor(num_frames_per_sample / 5)).astype(np.int)
    axes1.set_xticks([5 * i for i in range(0, A)])
    axes1.set_xticklabels(['{:.3f}s'.format(i*5*time_end/num_frames_per_sample) for i in np.arange(0, A)],rotation=-30,fontsize='small')
    axes1.set_ylabel('MFCCs')
    axes1.set_xlabel('Time [sec]')
    axes1.set_title('MFCCs heatmap')



if __name__ == '__main__':
    ################ raw data dir:
    data_file = r'/home/brutal/PycharmProjects/dataset-lunsound/ICBHI_final_database_data_preprocess/wave_data.DAT'
    label_file = r'/home/brutal/PycharmProjects/dataset-lunsound/ICBHI_final_database_data_preprocess/label.DAT'
    des_file = r'/home/brutal/PycharmProjects/dataset-lunsound/ICBHI_final_database_data_preprocess/des.DAT'

    # record_file = 'ICBHI_processed_data/Wave_MFCCs.tfrecords'
    # record_file = 'ICBHI_processed_data/Wave.tfrecords'
    # record_file = 'ICBHI_processed_data/Wave_MFCCs_more_detail.tfrecords'
    # record_file = 'ICBHI_processed_data/Wave_MFCCs_training.tfrecords'
    record_file = '/home/brutal/PycharmProjects/dataset-lunsound/ICBHI_processed_data/Wave_MFCCs_testing.tfrecords'
    ################ write into TFrecord:
    Fs=8000
    process_raw_data(des_file, data_file, label_file, Fs, record_file)
#########################
    # wave_data_dict, label_dicts, wave_num_cycle_dict,wave_data_max_len=process_raw_data(des_file, data_file, label_file, Fs,record_file)
    # print(wave_data_max_len)
    # print(wave_num_cycle_dict)
    # print(label_dicts)
    # for k, k2, k3 in zip(wave_data_dict,label_dicts,wave_num_cycle_dict):#,您将访问键而不是值
    #     num = wave_num_cycle_dict[k3]
    #     print('efid_wave',k)
    #     print('efid_label', k2)
    #     print('efid_wave_num_cycle_dict', k3)
    #     print('wave_num_cycle_dict', wave_num_cycle_dict[k3])
    #
    #     for i in range(0,wave_num_cycle_dict[k3]):
    #         print('wave', wave_data_dict[k][i])
    #         print('label', label_dicts[k2][i])
    #         print('wave type', type(wave_data_dict[k][i]))
    #         print('label type', type(label_dicts[k2][i]))
#########################





















