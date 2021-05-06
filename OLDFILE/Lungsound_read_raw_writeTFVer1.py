import tensorflow as tf
import numpy as np
import pandas as pd


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
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) # becareful!!!!!!!!!!!!!!about[A] and A

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
    fig = plt.figure(2,figsize=(25, 5),dpi=200) #开启一个窗口，同时设置大小，分辨率
    axes1 = fig.add_subplot(1, 1, 1) #通过fig添加子图，参数：行数，列数，第几个。
    axes1.plot(time_x_axis,wave_data, 'r')
    # plt.plot(np.arange(max_f.shape[0]), max_f))
    axes1.set_ylabel('SoundWave Amplitude')
    axes1.set_xlabel('Time [sec]')
    axes1.set_title('SoundWave Figure')
    plt.show()

def wave_resampled(wave_data, Fs_pre, Fs_after):
    time_total=len(wave_data)//Fs_pre
    x_before=np.linspace(0,time_total,len(wave_data))
    x_after = np.linspace(0, time_total, time_total*Fs_after)
    wave_data_resampled = np.interp(x_after, x_before, wave_data)
    return wave_data_resampled


def process_raw_data(des_file, data_file, label_file, Fs, record_file):
    piece_len = 20
    file_len = os.path.getsize(des_file)
    Num_Wave_frames=file_len // piece_len
    print('The total number of wave data in database={}'.format(Num_Wave_frames))

    with tf.io.TFRecordWriter(record_file) as writer:
        with open(des_file, 'rb') as data_handler:
            label_dicts = {}  #:dict be careful
            wave_data_dict = {}  #:dict be careful
            wave_num_cycle_dict = {}
            wave_data_max_len = []
            Fs_after = Fs
            for i in range(Num_Wave_frames):  # total of Num_Wave_frames, each Num_Wave_frames contains different num_of_cycle
                if i > Num_Wave_frames:
                # if i > 50:
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
                # plot_sound_wave(wave_data, framerate)
                # plot_sound_wave(wave_data_np, framerate)
                # plot_sound_wave(wave_after_resampled, Fs_after)
                # print('wave_data={}'.format(len(wave_after_resampled)))
                # print('framerate={}'.format(framerate))
                # print('framerate_after={}'.format(Fs_after))
                # print('len_label={}'.format(len_label))
                # print('labels={}'.format(labels))
                # print('data_strat_end={}'.format(data_strat_end))

                label_arr = np.asarray(labels).reshape(-1, 2).tolist()  # 将数组或者矩阵转换成列表
                label_dicts[efid] = label_arr
                data_strat_end_point = np.asarray(data_strat_end).reshape(-1, 2) * Fs_after

                data_segs = []
                data_segs_len = []
                num_of_lungsound_cycle = len(data_strat_end_point)

                # print('labels after processed={}'.format(label_dicts))
                # print('data_strat_end={}'.format(data_strat_end_point))
                # print('num_of_lungsound_cycle={}'.format(num_of_lungsound_cycle))

                for i in range(num_of_lungsound_cycle):
                    start_point = np.floor(data_strat_end_point[i][0]).astype(np.uint32)
                    end_point = np.ceil(data_strat_end_point[i][1]).astype(np.uint32)
                    data_seg = np.array(wave_after_resampled[start_point:end_point])

                    # print('start_point:', start_point)
                    # print('end_point:', end_point)
                    # print('Length of data_seg=          {}'.format(len(data_seg)))

                    # print('data_seg:', data_seg)
                    data_segs.append(data_seg)
                    data_segs_len.append(len(data_seg))
                wave_data_dict[efid] = data_segs
                wave_num_cycle_dict[efid] = num_of_lungsound_cycle
                wave_data_max_len.append(np.max(data_segs_len))
                # print('wave_data processed={}'.format(wave_data_pairs))
                # print('wave_data_max_len={}'.format(wave_data_max_len))
                for k, k2, k3 in zip(wave_data_dict, label_dicts, wave_num_cycle_dict):
                    num = wave_num_cycle_dict[k3]
                    for i in range(num):
                        efid_index_cycle=np.asarray([k,i],dtype=np.uint32)
                        # print(efid_index_cycle)
                        # example = serialize_data(efid_index_cycle, wave_data_dict[k][i], label_dicts[k2][i])
                        example = serialize_data(efid_index_cycle, wave_data_dict[k][i], efid_index_cycle)
                        writer.write(example)


def process_raw_data(des_file, data_file, label_file, Fs, record_file):
    piece_len = 20
    file_len = os.path.getsize(des_file)
    Num_Wave_frames = file_len // piece_len
    print('The total number of wave data in database={}'.format(Num_Wave_frames))

    with open(des_file, 'rb') as data_handler:
        label_dicts = {}  #:dict be careful
        wave_data_dict = {}  #:dict be careful
        wave_num_cycle_dict = {}
        wave_data_max_len = []
        Fs_after = Fs
        for i in range(
                Num_Wave_frames):  # total of Num_Wave_frames, each Num_Wave_frames contains different num_of_cycle
            # if i > Num_Wave_frames:
            if i > 2:
                break
            data = data_handler.read(piece_len)
            efid, = struct.unpack('<I', data[:4])
            seek_pos_wave, = struct.unpack('<I', data[4:8])
            piece_len_wave, = struct.unpack('<I', data[8:12])
            seek_pos_label, = struct.unpack('<I', data[12:16])
            piece_len_label, = struct.unpack('<I', data[16:20])

            wave_data = get_data(data_file, seek_pos_wave, piece_len_wave)
            framerate, len_label, labels, data_strat_end = view_label(label_file, seek_pos_label, piece_len_label)
            wave_data_np = np.asarray(wave_data)

            wave_after_resampled = wave_resampled(wave_data, framerate, Fs_after)

            # plot_sound_wave(wave_data, framerate)
            # plot_sound_wave(wave_data_np, framerate)
            # plot_sound_wave(wave_after_resampled, Fs_after)
            # print('wave_data={}'.format(len(wave_after_resampled)))
            # print('framerate={}'.format(framerate))
            # print('framerate_after={}'.format(Fs_after))
            # print('len_label={}'.format(len_label))
            # print('labels={}'.format(labels))
            # print('data_strat_end={}'.format(data_strat_end))

            # data_preprocess(efid, wave_data, framerate, labels, data_strat_end, label_pairs, wave_data_pairs)
            label_arr = np.asarray(labels).reshape(-1, 2).tolist()  # 将数组或者矩阵转换成列表
            label_dicts[efid] = label_arr
            data_strat_end_point = np.asarray(data_strat_end).reshape(-1, 2) * Fs_after

            data_segs = []
            data_segs_len = []
            num_of_lungsound_cycle = len(data_strat_end_point)

            # print('labels after processed={}'.format(label_dicts))
            # print('data_strat_end={}'.format(data_strat_end_point))
            # print('num_of_lungsound_cycle={}'.format(num_of_lungsound_cycle))

            for i in range(num_of_lungsound_cycle):
                start_point = np.floor(data_strat_end_point[i][0]).astype(np.uint32)
                end_point = np.ceil(data_strat_end_point[i][1]).astype(np.uint32)
                data_seg = np.array(wave_after_resampled[start_point:end_point])

                # print('start_point:', start_point)
                # print('end_point:', end_point)
                # print('Length of data_seg=          {}'.format(len(data_seg)))

                # print('data_seg:', data_seg)
                data_segs.append(data_seg)
                data_segs_len.append(len(data_seg))
            wave_data_dict[efid] = data_segs
            wave_num_cycle_dict[efid] = num_of_lungsound_cycle
            wave_data_max_len.append(np.max(data_segs_len))
            # print('wave_data processed={}'.format(wave_data_pairs))
            # print('wave_data_max_len={}'.format(wave_data_max_len))
    return wave_data_dict, label_dicts, wave_num_cycle_dict, wave_data_max_len


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



if __name__ == '__main__':
    ################ raw data dir:
    data_file = r'/home/brutal/PycharmProjects/Project-tf2.3/ICBHI_final_database_data_preprocess/wave_data.DAT'
    label_file = r'/home/brutal/PycharmProjects/Project-tf2.3/ICBHI_final_database_data_preprocess/label.DAT'
    des_file = r'/home/brutal/PycharmProjects/Project-tf2.3/ICBHI_final_database_data_preprocess//des.DAT'

    record_file = 'ICBHI_processed_data/Wave.tfrecords'
    ################ write into TFrecord:
    Fs=10000
    process_raw_data(des_file, data_file, label_file, Fs, record_file)
#########################
    wave_data_dict, label_dicts, wave_num_cycle_dict,wave_data_max_len=process_raw_data(des_file, data_file, label_file, Fs,record_file)
    print(wave_data_max_len)
    print(wave_num_cycle_dict)
    print(label_dicts)
    for k, k2, k3 in zip(wave_data_dict,label_dicts,wave_num_cycle_dict):#,您将访问键而不是值
        num = wave_num_cycle_dict[k3]
        print('efid_wave',k)
        print('efid_label', k2)
        print('efid_wave_num_cycle_dict', k3)
        print('wave_num_cycle_dict', wave_num_cycle_dict[k3])

        for i in range(0,wave_num_cycle_dict[k3]):
            print('wave', wave_data_dict[k][i])
            print('label', label_dicts[k2][i])
            print('wave type', type(wave_data_dict[k][i]))
            print('label type', type(label_dicts[k2][i]))
#########################





















