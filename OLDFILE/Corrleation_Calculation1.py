import os
import struct
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.signal
from scipy.fftpack import fft,ifft
#######################################################
FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', 'r/home/brutal/PycharmProjects/Digital Signal Process/data_QX/', """Path to the CIFAR-10 data directory.""")


def MSE(y, x):
    return np.sum(np.power((y - x),2))/len(y)

def Corrleation_after_norm(data_file):
	piece_len = 20004
	file_len = os.path.getsize(data_file)
	Number_of_Samples = file_len // piece_len
	# N1=np.floor((file_len // piece_len)*0.5)
	# Number_of_Samples=N1.astype(int)
	print('Number_of_Samples is {}'.format(Number_of_Samples))
	table_Corr_collected=np.zeros([2,Number_of_Samples])
	# table_Corr_collected = np.append()
	Sample_start_index=0
	print(file_len)
	with open(data_file, 'rb') as data_handler:
		data_handler.seek(piece_len * Sample_start_index, 0)
		for i in range(Number_of_Samples):
			if i > 5:
			# if i > Number_of_Samples:# number of sample started at postion data_handler.seek
				break
			datas = data_handler.read(piece_len)
			efid, = struct.unpack('<I', datas[:4])
			print('Its efid is {0} and current sample index:{1}'.format(efid,i))
			ecg_original_data_non_normliazed = struct.unpack('<2500f', datas[4:10004])#一个点4字节共2500个点
			ecg_simulated_data_non_normliazed = struct.unpack('<2500f', datas[10004:piece_len])

			##################### Specturm
			fft_simu_non_normlized=fft(ecg_original_data_non_normliazed)/len(ecg_original_data_non_normliazed)
			fft_orig_non_normlized=fft(ecg_simulated_data_non_normliazed)/len(ecg_simulated_data_non_normliazed)
			A=np.abs(fft_simu_non_normlized)[0:1250]
			B=np.abs(fft_orig_non_normlized)[0:1250]
			# plot_0(A, B)


			ecg_original_data = (ecg_original_data_non_normliazed-np.min(ecg_original_data_non_normliazed))/(np.max(ecg_original_data_non_normliazed)-np.min(ecg_original_data_non_normliazed))
			ecg_simulated_data = (ecg_simulated_data_non_normliazed-np.min(ecg_simulated_data_non_normliazed))/(np.max(ecg_simulated_data_non_normliazed)-np.min(ecg_simulated_data_non_normliazed))
			############ corrleation ###########
			Corr_Ori_simul=scipy.signal.correlate(ecg_original_data, ecg_simulated_data, 'same')
			Corr_Ori_simul_norm=np.max(Corr_Ori_simul)-np.min(Corr_Ori_simul)
			Corr_Ori_simul_norm1 = np.linalg.norm(Corr_Ori_simul, ord=np.inf)
			# Corr_simul_Ori = scipy.signal.correlate(ecg_simulated_data, ecg_original_data, 'same')
			# Corr_simul_Ori_norm = np.max(Corr_simul_Ori)-np.min(Corr_simul_Ori)
			# Corr_simul_Ori_norm1 = np.linalg.norm(Corr_simul_Ori, ord=np.inf)
			############ MSE ###########
			MSE_Ori_simul=MSE(ecg_simulated_data,ecg_original_data)







			# plot_0(Corr_Ori_simul/Corr_Ori_simul_norm,Corr_simul_Ori/Corr_simul_Ori_norm)
			# print(len(Corr_simul_Ori),len(Corr_Ori_simul))
			print('Its corrleation is {} and Its MSE is {}'.format(Corr_Ori_simul_norm, MSE_Ori_simul))
			table_Corr_collected[0,i]=Corr_Ori_simul_norm1
			table_Corr_collected[1,i] = MSE_Ori_simul
			# table_Corr_collected[1,i]=Corr_simul_Ori_norm1
		plot_1(table_Corr_collected[0,:], table_Corr_collected[1,:],Number_of_Samples)
		Most_like_index=[np.argmax(table_Corr_collected[0,:]), np.argmax(table_Corr_collected[1,:])]
		Most_unlike_index=[np.argmin(table_Corr_collected[0,:]), np.argmin(table_Corr_collected[1,:])]
		print(Most_like_index,Most_unlike_index)



		with open(data_file, 'rb') as data_handler:
			for j in range(2):
				if j==0:
					Type='Correlation'
				else:
					Type = 'MSE'
				data_handler.seek(piece_len * Most_like_index[j], 0)
				########## The most like figure ################
				for i in range(Number_of_Samples):
					if i > 0:  # number of sample started at postion data_handler.seek
						break
					datas = data_handler.read(piece_len)
					efid, = struct.unpack('<I', datas[:4])
					print('Its efid is {0} and current sample index:{1}'.format(efid,Most_like_index[j]+i))
					ecg_original_data = struct.unpack('<2500f', datas[4:10004])#一个点4字节共2500个点
					ecg_simulated_data = struct.unpack('<2500f', datas[10004:piece_len])
					plot_8s(ecg_original_data, ecg_simulated_data,'Type: {} Most likely'.format(Type))

				########## The most lunike figure ################
				data_handler.seek(piece_len * Most_unlike_index[j], 0)
				for i in range(Number_of_Samples):
					if i > 0:  # number of sample started at postion data_handler.seek
						break
					datas = data_handler.read(piece_len)
					efid, = struct.unpack('<I', datas[:4])
					print('Its efid is {0} and current sample index:{1}'.format(efid, Most_unlike_index[j]+i))
					ecg_original_data = struct.unpack('<2500f', datas[4:10004])  # 一个点4字节共2500个点
					ecg_simulated_data = struct.unpack('<2500f', datas[10004:piece_len])
					plot_8s(ecg_original_data, ecg_simulated_data,'Type: {} Most Unlikely'.format(Type))

# def plot_0(data1, data2):
# 	# xdata=np.arange(0,Number_of_x)
# 	fig = plt.figure(figsize=(25, 10))
#
# 	axes1 = fig.add_subplot(2, 1, 1)
# 	axes2 = fig.add_subplot(2, 1, 2)
#
# 	# bar_width = 0.35
# 	axes1.plot(data1, 'b-')
# 	# axes1.bar(xdata,data1, color='b',width=bar_width)
# 	# miloc_x1 = plt.MultipleLocator(1) #
# 	# miloc_y1 = plt.MultipleLocator(50) #
# 	# axes1.xaxis.set_major_locator(miloc_x1)
# 	# axes1.yaxis.set_major_locator(miloc_y1)
# 	# axes1.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)
#
#
# 	axes2.plot(data2, 'r-')
# 	# axes2.bar(xdata,data2, color='r',width=bar_width)
# 	# axes2.xaxis.set_major_locator(miloc_x1)
# 	# axes2.yaxis.set_major_locator(miloc_y1)
# 	# axes2.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)
#
#
#
# 	plt.show()
#
# def plot_1(data1, data2, Number_of_x):
# 	xdata=np.arange(0,Number_of_x)
# 	fig = plt.figure(figsize=(25, 10))
#
# 	axes1 = fig.add_subplot(2, 1, 1)
# 	axes2 = fig.add_subplot(2, 1, 2)
#
# 	bar_width = 0.35
# 	# axes1.plot(data1, 'b-')
# 	axes1.bar(xdata,data1/np.max(data1), color='b',width=bar_width)
# 	miloc_x1 = plt.MultipleLocator(1) #
# 	miloc_y1 = plt.MultipleLocator(0.05) #
# 	axes1.xaxis.set_major_locator(miloc_x1)
# 	axes1.yaxis.set_major_locator(miloc_y1)
# 	axes1.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)
#
#
# 	# axes2.plot(data2, 'r-')
# 	axes2.bar(xdata,data2/np.max(data2), color='r',width=bar_width)
# 	axes2.xaxis.set_major_locator(miloc_x1)
# 	axes2.yaxis.set_major_locator(miloc_y1)
# 	axes2.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)
#
#
#
# 	plt.show()

def process_axes(axes, datas, ld_label):
	axes.plot(datas, c='k', lw='0.8')

	axes.set_xlim(0, 2500)
	axes.set_ylim(-2.5, 2.5)

	miloc_x1 = plt.MultipleLocator(50) #每50一个中格，共有2500/50=50个中格
	miloc_y1 = plt.MultipleLocator(0.5) #每0.5毫伏 一个中格
	axes.xaxis.set_minor_locator(miloc_x1)
	axes.yaxis.set_minor_locator(miloc_y1)
	axes.grid(axis='both', which='major', c='r', ls='-', lw='0.5')

	miloc_x2 = plt.MultipleLocator(10) #10个点一个小格，一共有50/10=5小格
	miloc_y2 = plt.MultipleLocator(0.1) #0.5/5 = 0.1 一个小格
	axes.xaxis.set_minor_locator(miloc_x2)
	axes.yaxis.set_minor_locator(miloc_y2)
	axes.grid(axis='both', which='minor', c='#ff7855', ls=':', lw='0.5')

	axes.set_xticks([50 * i for i in range(0, (1 + 50))])
	axes.set_yticks([(0.5 * i) for i in range(-5, 6)])
	axes.set_xticklabels(['%s' % (i * 50) if i % 2 == 0 else '' for i in range(1 + 50)])
	axes.set_yticklabels(['%smv' % (0.5 * i) for i in range(-5, 6)])
	axes.legend(['{}'.format(ld_label)],loc=4)
	# 'best'         : 0, (only implemented for axes legends)(自适应方式)
	# 'upper right'  : 1,
	# 'upper left'   : 2,
	# 'lower left'   : 3,
	# 'lower right'  : 4,
	# 'right'        : 5,
	# 'center left'  : 6,
	# 'center right' : 7,
	# 'lower center' : 8,
	# 'upper center' : 9,
	# 'center'       : 10,


def plot_10s(fig, original_data, number_of_files, index, label):
	axes = fig.add_subplot(number_of_files, 1, index+1)
	# axes = fig.add_subplot(1, 1, 1)

	process_axes(axes, original_data, label)
	# plt.show()



def Write_original_with_noise(data_file, number_of_files):
	NF=number_of_files+1
	efid_len=4
	Each_piece_fraction=10000
	print('Number_of_Files is {}'.format(NF-1))
	Piece_len_Final=efid_len+NF*Each_piece_fraction
	write_file = r'/home/brutal/PycharmProjects/Digital Signal Process/data_QX/64_integral.dat'
	file_total_len=os.path.getsize(data_file[0])
	Number_of_Samples = file_total_len // 20004
	print('Number_of_Samples in each file = {}'.format(Number_of_Samples))

	with open(write_file, 'wb') as write_handler:
		for i in range(Number_of_Samples):
			write_handler.seek(Piece_len_Final * i, 0)
			for f in data_file:
				file_len = os.path.getsize(f)
				# print(file_len)
				if file_len == 1280256:
					piece_len = 20004
				else:
					piece_len = 10004  # QX_data
				with open(f, 'rb') as Read_data_handler:
					Read_data_handler.seek(piece_len * i, 0)
					datas = Read_data_handler.read(piece_len)
					if piece_len == 20004:
						write_handler.write(datas)
					else:
						datas_2=datas[4:10004]
						write_handler.write(datas_2)
	print('Writing is done')


def view_data(data_file,number_of_files):
	NF=number_of_files+1
	efid_len=4
	Each_piece_fraction=10000
	piece_len=efid_len+NF*Each_piece_fraction

	file_len = os.path.getsize(data_file)
	Number_of_Samples = file_len // piece_len
	print('Number_of_Samples is {}'.format(Number_of_Samples))
	file_len = os.path.getsize(data_file)
	print('The bytes size={}'.format(file_len))
	with open(data_file, 'rb') as data_handler:
		data_handler.seek(piece_len * 10, 0)
		# data_handler = open(data_file, 'rb')
		for j in range(file_len // piece_len):
			if j >=1:
				break
			datas = data_handler.read(piece_len)
			efid = struct.unpack('<I', datas[:4])
			print('efid={}'.format(efid))
			fig1 = plt.figure(figsize=(20, 10))
			fig2 = plt.figure(figsize=(20, 10))
			fig3 = plt.figure(figsize=(20, 10))
			fig4 = plt.figure(figsize=(20, 10))
			for i in range(NF):
				original_data = struct.unpack('<' + '2500f',datas[4 + Each_piece_fraction * i:4 + Each_piece_fraction * (i + 1)])
				# spread=int(np.ceil(NF/2))
				if i<2:
					plot_10s(fig1, original_data, 2, i, 'Data_{}0'.format(i))
				elif 4>i>=2:
					plot_10s(fig2, original_data, 2, i-2, 'Data_{}0'.format(i))
				elif 6>i>=4:
					plot_10s(fig3, original_data, 2, i - 4, 'Data_{}0'.format(i))
				else:
					plot_10s(fig4, original_data, 2, i - 6, 'Data_{}0'.format(i))
			plt.show()#必须全部绘制完毕才能plt.show()

	print('done!!!')









if __name__ == '__main__':
	data_dir = r'/home/brutal/PycharmProjects/Digital Signal Process/data_QX/'
	data_dir1 = r'/home/brutal/PycharmProjects/Digital Signal Process/data_QX/64_integral.dat'
	filenames = [os.path.join(data_dir, '64.dat')]
	filenames1 = [os.path.join(data_dir, 'emg_{:d}.dat'.format(i)) for i in np.arange(30,51,10)]
	filenames2 = [os.path.join(data_dir, 'gauss_{:d}.dat'.format(i)) for i in np.arange(30,51,10)]

	for i in np.arange(1,3,1):
		A='filenames.extend(filenames{:d})'.format(i)
		exec(A)

	for f in filenames:
		if not tf.io.gfile.exists(f):
			raise ValueError('Failed to find file: ' + f)
	# print(['The files below has been loaded: {}\n'.format(filenames[i]) for i in np.arange(0,7)])
	print('The files below has been loaded: \n {} '.format(filenames))
	print('The number of files={:d}'.format(len(filenames)))
	# Write_original_with_noise(filenames, len(filenames))
	view_data(data_dir1,len(filenames))

	# Corrleation_after_norm(filenames)


























