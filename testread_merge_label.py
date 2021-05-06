import struct
import os
import sys
from matplotlib import pyplot as plt
sys.platform = 'win64'


#读取信号数据
def view_data(data_file):
	piece_len = 640004
	data_handler = open(data_file, 'rb')
	file_len = os.path.getsize(data_file)
	# print('file_len', file_len)
	wave_data_list=[]
	for i in range(file_len//piece_len):
		if i >2:
			break
		data_handler.seek(i*piece_len)
		data = data_handler.read(piece_len)
		id, = struct.unpack('<I', data[:4])
		# print('id', id)
		wave_data = list(struct.unpack('<' + 'f' * 160000, data[4:640004]))
		# plt_wav(wave_data, id)
		wave_data_list.append(wave_data)
	data_handler.close()
	return  wave_data_list


#读取label数据
def view_label(label_file):
	piece_len = 12
	data_handler = open(label_file, 'rb')
	file_len = os.path.getsize(label_file)
	label_list = []
	print('file_len', file_len)
	for i in range(file_len // piece_len):
		if i >2:
			break
		data_handler.seek(i*piece_len)
		data = data_handler.read(piece_len)
		id, = struct.unpack('<I', data[:4])
		labels = struct.unpack('<' + 'I' * 2, data[4:])
		print('id', id)
		print('labels', labels)
		label_list.append(labels)
	data_handler.close()

	return label_list


def plt_wav(wave_data, id):
	fig = plt.figure(figsize=(18, 4))
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.plot(wave_data, color='b')
	plt.title("wav data: %s" % id)
	plt.xlabel("Time(s)")
	plt.ylabel("Amplitude")
	plt.show()



if __name__ == '__main__':
	data_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210107/wave_data.DAT'
	label_file = r'/home/brutal/PycharmProjects/Data_base/lung_sound_merge_label20210107/label.DAT'

	wave_list=view_data(data_file)
	label_list=view_label(label_file)
	# print(wave_list)
	# print(label_list)
