import soundfile as sf
import glob
import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import pandas as pd
import wave
import csv
import sys

# sys.platform = 'win64'

#读取数据描述文件
def describe_data(des_file, data_file, label_file):
	piece_len = 20
	file_len = os.path.getsize(des_file)
	Num_Wave_frames=file_len // piece_len
	print('The total number of wave data in database={}'.format(Num_Wave_frames))
	data_handler = open(des_file, 'rb')

	label_pairs = {}
	wave_data_pairs = {}
	for i in range(Num_Wave_frames):
		# if i > Num_Wave_frames:
		if i >1:
			break
		data = data_handler.read(piece_len)
		efid, = struct.unpack('<I', data[:4])
		seek_pos_wave, = struct.unpack('<I', data[4:8])
		piece_len_wave, = struct.unpack('<I', data[8:12])
		seek_pos_label, = struct.unpack('<I', data[12:16])
		piece_len_label, = struct.unpack('<I', data[16:20])

		# print('efid', efid)
		# print('seek_pos_wave', seek_pos_wave)
		# print('piece_len_wave', piece_len_wave)
		# print('seek_pos_label', seek_pos_label)
		# print('piece_len_label', piece_len_label)

		efid_wave, wave_data = view_data(data_file, seek_pos_wave, piece_len_wave)
		# print('wave_data type',type(wave_data[0]))
		efid_label, framerate, len_label, labels, data_strat_end = view_label(label_file, seek_pos_label,piece_len_label)
		# print(type(labels[0]))
		# print('label is like={}'.format(labels))
		print('data_strat_end={}'.format(data_strat_end))

		data_preprocess(efid, wave_data, framerate, labels, data_strat_end, label_pairs, wave_data_pairs)
		print('efid={}\nframerate={}\ndata_strat_end={}'.format(efid, framerate, data_strat_end))
		print('label_pairs:', len(label_pairs[0]), label_pairs[0], label_pairs.keys())
		print('wave_data_pairs:', len(wave_data_pairs[0]), len(wave_data_pairs[0][0]), wave_data_pairs.keys())



	data_handler.close()
	return label_pairs, wave_data_pairs

#读取信号数据
def view_data(data_file, seek_pos_wave, piece_len_wave):
	with open(data_file, 'rb') as data_handler:
		data_handler.seek(seek_pos_wave)
		data = data_handler.read(piece_len_wave)
		efid_wave, = struct.unpack('<I', data[:4])
		# print('efid_wave', efid_wave)
		len_wave_data, = struct.unpack('<I', data[4:8])
		print('len_wave_data',len_wave_data)
		# print('len_wave_data', len_wave_data)
		wave_data = struct.unpack('<' + 'f' * len_wave_data, data[8:])
	return efid_wave, wave_data

#读取label数据
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

		# print('efid_label', efid_label)
		# print('framerate', framerate)
		# print('len_label', len_label)
		# print('labels', labels)
		# print('data_strat_end', data_strat_end)

	return efid_label, framerate, len_label, labels, data_strat_end

#数据预处理
def data_preprocess(efid, wave_data, framerate, labels, data_strat_end, label_pairs, wave_data_pairs):
	data_segs = []
	label_arr = np.asarray(labels).reshape(-1, 2).tolist()#将数组或者矩阵转换成列表
	label_pairs[efid] = label_arr

	data_strat_end_point = np.asarray(data_strat_end).reshape(-1, 2) * framerate
	# print('data_strat_end_point', data_strat_end_point)

	for i in range(len(data_strat_end_point)):
		start_point = int(data_strat_end_point[i][0])
		end_point = int(data_strat_end_point[i][1])
		data_seg = list(wave_data[start_point:end_point])
		data_segs.append(data_seg)

	wave_data_pairs[efid] = data_segs
	# print(type(wave_data_pairs))


if __name__ == '__main__':
	data_file = r'/home/brutal/PycharmProjects/Project-tf2.3/ICBHI_final_database_data_preprocess/wave_data.DAT'
	label_file = r'/home/brutal/PycharmProjects/Project-tf2.3/ICBHI_final_database_data_preprocess/label.DAT'
	des_file = r'/home/brutal/PycharmProjects/Project-tf2.3/ICBHI_final_database_data_preprocess//des.DAT'

	describe_data(des_file, data_file, label_file)
