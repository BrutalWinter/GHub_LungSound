import struct
import os
import tensorflow as tf
from PulmonarySound_FuncBase import Calculating_MFCCs_from_wave, plt_wav_batch, plt_spectrum_from_wave_batchs, plt_MFCC_batch,plt_spectrogram_batch
###############################################
# 读取label数据
def view_label(label_file):
    piece_len = 12
    data_handler = open(label_file, 'rb')
    file_len = os.path.getsize(label_file)
    label_list = []
    # print('file_len', file_len)
    for i in range(file_len // piece_len):
        # if i > 2:
        #     break
        data_handler.seek(i * piece_len)
        data = data_handler.read(piece_len)
        id, = struct.unpack('<I', data[:4])
        labels = struct.unpack('<' + 'I' * 2, data[4:])
        label_list.append(labels)

        # print('id', id)
        # print('labels', labels)
    data_handler.close()
    return label_list

#读取信号数据
def view_data(data_file):
    piece_len = 640004
    data_handler = open(data_file, 'rb')
    file_len = os.path.getsize(data_file)
    # print('file_len', file_len)
    wave_data_list=[]
    for i in range(file_len//piece_len):
        # if i >2:
        #     break
        data_handler.seek(i*piece_len)
        data = data_handler.read(piece_len)
        id, = struct.unpack('<I', data[:4])
        wave_data = list(struct.unpack('<' + 'f' * 160000, data[4:640004]))
        wave_data_list.append(wave_data)

        # print('id', id)
        # plt_wav(wave_data, id)
    data_handler.close()
    return  wave_data_list









##########################################
##########################################
def Parsed_LungSound_label(data):

    raw_data_efid= tf.strings.substr(data, 0, 4)
    efid = tf.io.decode_raw(raw_data_efid, tf.int32)  ## corresponding I

    raw_label = tf.strings.substr(data, 4 + 4 * 0, 4 * 1)  ######## number of heartbeat
    label = tf.io.decode_raw(raw_label, tf.int32)  ## corresponding B

    label_merged=label
    pre_merged=tf.constant(3,tf.int32)
    post_merged=tf.constant(0,tf.int32)
    label_merged=tf.where(label_merged==pre_merged, x=post_merged, y=label_merged)

    pre_merged=tf.constant(4,tf.int32)
    post_merged=tf.constant(0,tf.int32)
    label_merged=tf.where(label_merged==pre_merged, x=post_merged, y=label_merged)

    return efid, label_merged

def Parsed_LungSound_data(data):

    raw_data_efid= tf.strings.substr(data, 0, 4)
    efid = tf.io.decode_raw(raw_data_efid, tf.int32)  ## corresponding I

    raw_data = tf.strings.substr(data, 4 + 4 * 0, 4 * 160000)  ######## number of heartbeat
    data = tf.io.decode_raw(raw_data, tf.float32)  ## corresponding B

    return efid, data


def label_all_0_keep(Labels_dataset_parsed,Dataset_parsed): #predicate:	A function mapping a dataset element to a boolean.
    A=tf.math.reduce_sum(Labels_dataset_parsed[1])
    return tf.math.less(A,1)

def label_all_1_keep(Labels_dataset_parsed,Dataset_parsed): #predicate:	A function mapping a dataset element to a boolean.
    A=tf.math.reduce_sum(Labels_dataset_parsed[1])
    return tf.math.equal(A,2)
##########################################
##########################################




if __name__ == '__main__':
    # data_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210107/wave_data.DAT'
    # label_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210107/label.DAT'

    # data_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_nanfang_selected/data_0.DAT'
    # label_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_nanfang_selected/label_0.DAT'

    ###########  ZN pipeline:
    # wave_list=view_data(data_file)
    # label_list=view_label(label_file)
    # print(wave_list)
    # print(label_list)
    ####################################
    Directory = r'/home/brutal/PycharmProjects/Data_base/LUNGSOUND'
    Wave_data_files_path = os.path.join(Directory, 'data_*.DAT')
    Wave_data_files = tf.io.gfile.glob(Wave_data_files_path)

    Label_files_path = os.path.join(Directory, 'label_*.DAT')
    Label_files = tf.io.gfile.glob(Label_files_path)

    # Training_beat_files = sorted(Training_beat_files + Test_beat_files)
    # Training_data_files = sorted(Training_data_files + Test_data_files)
    Wave_data_files = sorted(Wave_data_files)
    Label_files = sorted(Label_files)

    print('Wave_data_files')
    for i in Wave_data_files:
        print(i)

    print('Training_data_file:')
    for i in Label_files:
        print(i)



    ############    tf pipeline:
    Label_dataset = tf.data.FixedLengthRecordDataset(filenames = Label_files, record_bytes = 8)
    Labels_dataset_parsed = Label_dataset.map(Parsed_LungSound_label,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Labels_Rpos_dataset_parsed = Label_Rpos_dataset.map(Parsed_heart_beat)

    Dataset = tf.data.FixedLengthRecordDataset(filenames=Wave_data_files, record_bytes=640004)
    Dataset_parsed = Dataset.map(Parsed_LungSound_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Ecg_dataset_parsed = Ecg_dataset.map(Parsed_Ecg_data)


    batch_size=2
    Label_Data_dataset=tf.data.Dataset.zip((Labels_dataset_parsed,Dataset_parsed)).batch(batch_size)
    # .shuffle(900)
    # Label_Data_dataset=Label_Data_dataset.filter(label_all_0_keep).batch(1)

    ##########################################
    start = 0
    end = 5

    for step, data in enumerate(Label_Data_dataset):
        if step > end + 1:
            break

        if step >= start and step <= end:
            id_batch = data[1][0]
            Label_batch = data[0][1]
            PCM_batch= data[1][1]
            print('==>The {:d}th -- its efid is {}, shape={}, dtype={}'.format(step, id_batch, id_batch.shape,id_batch.dtype))
            print('==>The {:d}th -- its label is {}, shape={}, dtype={}'.format(step, Label_batch,Label_batch.shape,Label_batch.dtype))
            print('==>The {:d}th -- its data is {}, shape={}, dtype={}'.format(step, PCM_batch.shape,PCM_batch.shape,PCM_batch.dtype))
            # print('label difference={}'.format(data[0][1]-label_list[step]))
            # print('data difference={}'.format(tf.math.reduce_sum(data[1][1] - wave_list[step])))


            # plt_wav_batch(PCM_batch, id_batch, Label_batch, sample_rate=8000)
            # plt_spectrum_from_wave_batchs(PCM_batch, id_batch, Label_batch, sample_rate=8000, FFT_len=1024 * 16, overlap=1024 * 16 - 256 * 16)

            ## 呼吸周期大概 2-3s
            spectrograms_batch, mfccs_batch=Calculating_MFCCs_from_wave(PCM_batch,
                                    sample_rate=8000, window_frame_len=1024*1, frame_step=256*1, fft_length=1024*1, num_mel_bins=130, num_mel_keepbins=30)
            #
            # print('spectrograms_batch',spectrograms_batch.shape)
            # print('mfccs_batch', mfccs_batch.shape)
            plt_spectrogram_batch(spectrograms_batch, id_batch, Label_batch, time_duration=20, sample_rate=8000)
            plt_MFCC_batch(mfccs_batch, id_batch, Label_batch, time_duration=20)
















