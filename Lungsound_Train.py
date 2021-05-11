import tensorflow as tf
import os
from PulmonarySound_FuncBase import Calculating_MFCCs_from_wave, \
    plt_wav_batch, plt_spectrum_from_wave_batchs, plt_MFCC_batch,plt_spectrogram_batch

from PulmonarySound_datapipeline import Parsed_LungSound_data, Parsed_LungSound_label
from Lung_sound_model import LungSound_Model, train_step, predict

###############################################





####################### Training Purpose
if __name__ == '__main__':
    ####################################
    Directory = r'/home/brutal/PycharmProjects/LUNGSOUND'
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
    Label_dataset = tf.data.FixedLengthRecordDataset(filenames=Label_files, record_bytes=8)
    Labels_dataset_parsed = Label_dataset.map(Parsed_LungSound_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    Dataset = tf.data.FixedLengthRecordDataset(filenames=Wave_data_files, record_bytes=640004)
    Dataset_parsed = Dataset.map(Parsed_LungSound_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    Label_Data_dataset = tf.data.Dataset.zip((Labels_dataset_parsed, Dataset_parsed))
    Label_Data_dataset = Label_Data_dataset.shuffle(1600,seed=8).take(1200)


    ##########################################
    batch_size = 1
    Label_Data_dataset =Label_Data_dataset.shuffle(1200, reshuffle_each_iteration=True).batch(batch_size)
    # Label_Data_dataset=Label_Data_dataset.filter(label_all_0_keep).batch(1)
    LungSound_model = LungSound_Model(32, 5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-08)  # >minimum=1e-6

    modelsave_directory= r"/home/brutal/PycharmProjects/Project-lungsound/Model_Save"


    start = 0
    end = 5
    for epoch in tf.range(0, 16):
        for step, data in enumerate(Label_Data_dataset):
            if step > end + 1:
                break
            #######################

            id_batch = data[1][0]
            Label_batch = data[0][1]
            PCM_batch = data[1][1]
            # True_label_batch=tf.math.reduce_min(Label_batch,axis=-1,keepdims=True)
            # ######################## print info:
            # print(step,':<========================>')
            # print('The {:d}th -- its efid is {}, shape={}, dtype={}'.format(step, id_batch, id_batch.shape,
            #                                                                    id_batch.dtype))
            # print('The {:d}th -- its label is {}, shape={}, dtype={}'.format(step, Label_batch, Label_batch.shape,
            #                                                                     Label_batch.dtype))
            # # print('==>The {:d}th -- its True_label_batch is {}, shape={}, dtype={}'.format(step, True_label_batch, True_label_batch.shape,
            # #                                                                     True_label_batch.dtype))
            # print('The {:d}th -- its data is {}, shape={}, dtype={}'.format(step, PCM_batch.shape, PCM_batch.shape,
            #                                                                    PCM_batch.dtype))
            #
            # plt_wav_batch(PCM_batch, id_batch, Label_batch, sample_rate=8000)

            # # plt_spectrum_from_wave_batchs(PCM_batch, id_batch, Label_batch, sample_rate=8000, FFT_len=1024 * 16, overlap=1024 * 16 - 256 * 16)





            ############################################################
            # ## 呼吸周期大概 2-3s
            Weight=10 ## MFCCs is 1
            spectrograms_batch, mfccs_batch = Calculating_MFCCs_from_wave(PCM_batch,
                                                                          sample_rate=8000, window_frame_len=1024 * Weight,
                                                                          frame_step=256 * Weight, fft_length=1024 * 1,
                                                                          num_mel_bins=120, num_mel_keepbins=30)

            print('spectrograms_batch', spectrograms_batch.shape)
            print('mfccs_batch', mfccs_batch.shape)

            plt_spectrogram_batch(spectrograms_batch, id_batch, Label_batch, time_duration=20, sample_rate=8000)
            plt_MFCC_batch(mfccs_batch, id_batch, Label_batch, time_duration=20)

            # mfccs_batch = tf.expand_dims(mfccs_batch, axis=-1)
            # loss = train_step(LungSound_model, optimizer, mfccs_batch, Label_batch)

            # spectrograms_batch = tf.expand_dims(spectrograms_batch, axis=-1)
            # loss = train_step(LungSound_model, optimizer, spectrograms_batch, Label_batch)

    #         print('The Current Epoch={} and its step={}'.format(epoch,step))
    #         print('loss', loss)
    #         print('\n<===============================>')
    #
    #
    #     if (epoch+1) % 4 == 0:
    #         print('==========>  Saving epoch end model  <==========')
    #         model_save_path = os.path.join(modelsave_directory,'LungModel_epoch_Spec_75percent{}'.format(epoch)) #_6layersoff
    #         tf.keras.models.save_model(LungSound_model, filepath=model_save_path)
    #
    # print('done')






# ######################## Predict Purpose
# if __name__ == '__main__':
#     ####################################
#     Directory = r'/home/brutal/PycharmProjects/LUNGSOUND'
#     Wave_data_files_path = os.path.join(Directory, 'data_*.DAT')
#     Wave_data_files = tf.io.gfile.glob(Wave_data_files_path)
#
#     Label_files_path = os.path.join(Directory, 'label_*.DAT')
#     Label_files = tf.io.gfile.glob(Label_files_path)
#
#     # Training_beat_files = sorted(Training_beat_files + Test_beat_files)
#     # Training_data_files = sorted(Training_data_files + Test_data_files)
#     Wave_data_files = sorted(Wave_data_files)
#     Label_files = sorted(Label_files)
#
#     print('Wave_data_files')
#     for i in Wave_data_files:
#         print(i)
#
#     print('Training_data_file:')
#     for i in Label_files:
#         print(i)
#
#     ############    tf pipeline:
#     Label_dataset = tf.data.FixedLengthRecordDataset(filenames=Label_files, record_bytes=8)
#     Labels_dataset_parsed = Label_dataset.map(Parsed_LungSound_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#
#     Dataset = tf.data.FixedLengthRecordDataset(filenames=Wave_data_files, record_bytes=640004)
#     Dataset_parsed = Dataset.map(Parsed_LungSound_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#
#     Label_Data_dataset = tf.data.Dataset.zip((Labels_dataset_parsed, Dataset_parsed))
#     Label_Data_dataset = Label_Data_dataset.shuffle(1600,seed=8).skip(1200)
#
#
#
#
#
#     ##########################################
#     batch_size = 8
#     Label_Data_dataset = Label_Data_dataset.batch(batch_size)
#     modelload_directory = r"/home/brutal/PycharmProjects/GHub_LungSound/Model_Save"
#
#     checkpoint_path_traning_pre = os.path.join(modelload_directory, 'LungModel_epoch_75percent11')
#     LungSound_model = tf.keras.models.load_model(filepath=checkpoint_path_traning_pre,compile=False)
#
#
#     modelsave_directory= r"/home/brutal/PycharmProjects/GHub_LungSound/Model_Save"
#
#     C_Matrix = tf.zeros([2, 2], dtype=tf.int32)
#
#     end = 10
#     for step, data in enumerate(Label_Data_dataset):
#         # if step > end + 1:
#         #     break
#         #######################
#
#         id_batch = data[1][0]
#         Label_batch = data[0][1]
#         PCM_batch = data[1][1]
#         # ######################## print info:
#         print('step=',step)
#
#         # print('The {:d}th -- its efid is {}, shape={}, dtype={}'.format(step, id_batch, id_batch.shape,
#         #                                                                    id_batch.dtype))
#         # print('The {:d}th -- its label is {}, shape={}, dtype={}'.format(step, Label_batch, Label_batch.shape,
#         #                                                                     Label_batch.dtype))
#         # print('The {:d}th -- its data is {}, shape={}, dtype={}'.format(step, PCM_batch.shape, PCM_batch.shape,
#         #                                                                    PCM_batch.dtype))
#
#         # plt_wav_batch(PCM_batch, id_batch, Label_batch, sample_rate=8000)
#
#         # plt_spectrum_from_wave_batchs(PCM_batch, id_batch, Label_batch, sample_rate=8000, FFT_len=1024 * 16, overlap=1024 * 16 - 256 * 16)
#
#         ############################################################
#         # ## 呼吸周期大概 2-3s
#         spectrograms_batch, mfccs_batch = Calculating_MFCCs_from_wave(PCM_batch,
#                                                                     sample_rate=8000, window_frame_len=1024 * 1,
#                                                                     frame_step=256 * 1, fft_length=1024 * 1,
#                                                                     num_mel_bins=120, num_mel_keepbins=30)
#         mfccs_batch = tf.expand_dims(mfccs_batch, axis=-1)
#         # mfccs_batch = tf.expand_dims(spectrograms_batch, axis=-1)
#         # print('spectrograms_batch', spectrograms_batch.shape)
#         print('mfccs_batch', mfccs_batch.shape)
#
#         label_predict_prob = predict(LungSound_model, mfccs_batch)
#         label_predict=tf.math.argmax(label_predict_prob,axis=-1, output_type=tf.int32)
#
#
#         ############################################################
#         labels=tf.reshape(Label_batch,[-1])
#         print('labels',labels)
#         CM_single = tf.math.confusion_matrix(labels, label_predict, num_classes=2)
#         C_Matrix = C_Matrix + CM_single
#         ############################################################
#
#
#         # print('label_predict_prob', label_predict_prob)
#         print('label_predict',label_predict)
#         print('<========================>')
#
#     sum_row = tf.math.reduce_sum(C_Matrix, axis=-1)
#     sum_row = tf.where(sum_row == 0, x=1, y=sum_row)
#     sum_column = tf.math.reduce_sum(C_Matrix, axis=0)
#     C_Matrix_diagonal = tf.linalg.diag_part(C_Matrix)
#
#
#     # sum_row_all=tf.constant([916, 686],dtype=tf.int32)
#     sum_row_all = tf.constant([232,170], dtype=tf.int32)
#     Recall_diagonal=100 * C_Matrix_diagonal / sum_row_all
#     Precision_diagonal = 100 * C_Matrix_diagonal / sum_column
#     f1_score = Precision_diagonal * Recall_diagonal * 2 / (Recall_diagonal + Precision_diagonal)
#
#     Recall_matrics=100 * C_Matrix / tf.reshape(sum_row_all,[2, 1])
#     Precision_matrics = 100 * C_Matrix / tf.reshape(sum_column,[2, 1])
#
#
#     # print(de)
#     print('sum_row:',sum_row)
#     print('sum_column:', sum_column)
#     # np.set_printoptions(precision=2,suppress=True)
#     print('C_Matrix=', C_Matrix)
#     print('C_Matrix_diagonal='.format(C_Matrix_diagonal))
#     print('Recall_matrics={}'.format(Recall_matrics))
#     print('Precision_matrics={}'.format(Precision_matrics))
#
#
#
#     print(checkpoint_path_traning_pre)
#     print('Recall_diagonal   ={}'.format(Recall_diagonal))
#     print('Precision_diagonal={}'.format(Precision_diagonal))
#     print('f1_score          ={}'.format(f1_score))
