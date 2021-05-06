import soundfile as sf
import glob
import numpy as np
import struct
import os
import pandas as pd
from matplotlib import pyplot as plt
import wave
import csv
import sys

sys.platform = 'win64'

#读取数据描述文件
def describe_data(des_file, data_file, label_file):
	piece_len = 20
	file_len = os.path.getsize(des_file)
	data_handler = open(des_file, 'rb')
	label_pairs = {}
	wave_data_pairs = {}
	for i in range(file_len // piece_len):
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
		efid_label, framerate, labels = view_label(label_file, seek_pos_label, piece_len_label)

	data_handler.close()
	return label_pairs, wave_data_pairs

#读取信号数据
def view_data(data_file, seek_pos_wave, piece_len_wave):
	data_handler = open(data_file, 'rb')
	data_handler.seek(seek_pos_wave)
	data = data_handler.read(piece_len_wave)
	efid_wave, = struct.unpack('<I', data[:4])
	# print('efid_wave', efid_wave)
	len_wave_data, = struct.unpack('<I', data[4:8])
	# print('len_wave_data', len_wave_data)
	wave_data = struct.unpack('<' + 'f' * len_wave_data, data[8:])

	'''
	fig = plt.figure(figsize=(30, 8))
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.plot(wave_data, color='b')
	plt.title("signal_original")
	ax1.set_title('lung sound plot')
	plt.xlabel("Time(s)")
	plt.ylabel("Amplitude")
	plt.show()
	data_handler.close()
	'''
	return efid_wave, wave_data

#读取label数据
def view_label(label_file, seek_pos_label, piece_len_label):
	data_handler = open(label_file, 'rb')
	data_handler.seek(seek_pos_label)
	data = data_handler.read(piece_len_label)
	efid_label, = struct.unpack('<I', data[:4])
	framerate, = struct.unpack('<I', data[4:8])
	labels = struct.unpack('<' + 'I' * 2, data[8:])

	# print('efid_label', efid_label)
	# print('framerate', framerate)
	# print('labels', labels)

	data_handler.close()
	return efid_label, framerate, labels

#数据预处理
def data_preprocess(efid, wave_data, framerate, labels, data_strat_end, label_pairs, wave_data_pairs):
	data_segs = []
	label_arr = np.asarray(labels).reshape(-1, 2).tolist()
	print('label_arr', label_arr)
	label_pairs[efid] = label_arr
	print('label_pairs', label_pairs)
	data_strat_end_point = np.asarray(data_strat_end).reshape(-1, 2) * framerate
	# print('data_strat_end_point', data_strat_end_point)

	for i in range(len(data_strat_end_point)):
		if i == 9:
			start_point = int(data_strat_end_point[i][0])
			end_point = int(data_strat_end_point[i][1])
			data_seg = list(wave_data[start_point:end_point])
			data_segs.append(data_seg)
			print('start_point:end_point',(start_point,end_point ))
			print('data seg:', data_seg)

	# wave_data_pairs[efid] = data_segs
	# print('wave_data_pairs', wave_data_pairs)

	# print('label_pairs', len(label_pairs[108]), label_pairs[108])
	# print('wave_data_pairs', wave_data_pairs.keys(), len(wave_data_pairs[108]))


if __name__ == '__main__':
	data_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210106/wave_data.DAT'
	label_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210106/label.DAT'
	des_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210106/des.DAT'

	label_pairs, wave_data_pairs=describe_data(des_file, data_file, label_file)
