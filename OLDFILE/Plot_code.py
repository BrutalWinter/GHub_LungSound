import numpy as np
import matplotlib.pyplot as plt
############################################################
'''【Python】 【绘图】plt.figure()的使用'''
# x = np.arange(0, 100)
#
# plt.subplot(221)
# plt.plot(x, x)
#
# plt.subplot(222)
# plt.plot(x, -x)
#
# plt.subplot(223)
# plt.plot(x, x ** 2)
# plt.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
#
# plt.subplot(224)
# plt.plot(x, np.log(x))
# plt.show()
# ##################
# x = np.arange(0, 100)
#
# [fig,axes]=plt.subplots(2,2)
# ax1=axes[0,0]
# ax2=axes[0,1]
# ax3=axes[1,0]
# ax4=axes[1,1]
#
# ax1.plot(x, x)
#
# ax2.plot(x, -x)
#
# ax3.plot(x, x ** 2)
# ax3.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)
#
# ax4.plot(x, np.log(x))
# plt.show()
##################
x = np.arange(0, 100)

fig=plt.figure(figsize=(40,30),facecolor='white',edgecolor='green')
'''figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
num:图像编号或名称，数字为编号 ，字符串为名称
figsize:指定figure的宽和高，单位为英寸；
dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80      1英寸等于2.5cm,A4纸是 21*30cm的纸张
facecolor:背景颜色
edgecolor:边框颜色
frameon:是否显示边框'''

ax1=fig.add_subplot(2,2,1)
ax1.plot(x, x)

ax3=fig.add_subplot(2,2,3)
ax3.plot(x, x ** 2)
ax3.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)

ax4=fig.add_subplot(2,2,4)
ax4.plot(x, np.log(x))
plt.show()
##########################################################################################
fig = plt.figure(figsize=(20, 10))
# 定义数据
# x = [1, 2, 3, 4, 5, 6, 7]
# y = [1, 3, 4, 2, 5, 8, 6]
x=np.linspace(0,100,num=100)
y=np.linspace(-50,50,num=100)
#新建区域ax1

#figure的百分比,从figure 10%的位置开始绘制, 宽高是figure的80%
left, bottom, width, height = 0.01, 0.01, 0.8, 0.8
# 获得绘制的句柄
# ax1 = fig.add_axes([left, bottom, width, height])
ax1 = fig.add_subplot(1,1,1)
ax1.plot(x, y, 'g',label='legend1')
ax1.plot(x, y+10, 'r',label='legend2')
ax1.plot(x,y+15,marker='o',color='k',label='legend3')   #点图：marker图标
ax1.plot(x,y+20,linestyle='--',alpha=1,color='k',label='legend4')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

ax1.set_title('python-drawing')            #设置图体，plt.title
ax1.set_xlabel('x-name')                    #设置x轴名称,plt.xlabel
ax1.set_ylabel('y-name')                    #设置y轴名称,plt.ylabel
# plt.axis([-6,6,-10,10])                  #设置横纵坐标轴范围，这个在子图中被分解为下面两个函数
ax1.set_xlim(0,110)                           #设置横轴范围，会覆盖上面的横坐标,plt.xlim
ax1.set_ylim(-60,60)                         #设置纵轴范围，会覆盖上面的纵坐标,plt.ylim

xmajorLocator = plt.MultipleLocator(5)   #定义横向主刻度标签的刻度差为的倍数。就是隔几个刻度才显示一个标签文本
ymajorLocator = plt.MultipleLocator(10)   #定义纵向主刻度标签的刻度差为的倍数。就是隔几个刻度才显示一个标签文本
ax1.xaxis.set_major_locator(xmajorLocator) #x轴 应用定义的横向主刻度格式。如果不应用将采用默认刻度格式
ax1.yaxis.set_major_locator(ymajorLocator) #y轴 应用定义的纵向主刻度格式。如果不应用将采用默认刻度格式
# ax1.xaxis.grid(True, which='major',c='r', ls='-.', lw='2')      #x坐标轴的网格使用定义的主刻度格式
# ax1.yaxis.grid(True, which='major', c='r', ls='-', lw='0.5')      #x坐标轴的网格使用定义的主刻度格式
ax1.grid(axis='both', which='major', c='b', ls='-', lw='1')
xminorLocator = plt.MultipleLocator(1)
yminorLocator= plt.MultipleLocator(2)
ax1.xaxis.set_minor_locator(xminorLocator)
ax1.yaxis.set_minor_locator(yminorLocator)
# ax1.xaxis.grid(True, which='minor',c='b', ls='-.', lw='1')      #x坐标轴的网格使用定义的主刻度格式
# ax1.yaxis.grid(True, which='minor', c='b', ls='-', lw='0.25')      #x坐标轴的网格使用定义的主刻度格式
ax1.grid(axis='both', which='minor', c='r', ls=':', lw='0.5')
ax1.legend(loc='upper left')
# ax1.set_xticks([]) #去除坐标轴刻度
# x_new=np.linspace(-110,0,num=(22))#而在python3中， ‘整数/整数 = 浮点数’
ax1.set_xticks([1 * i for i in x])
# ax1.set_xticklabels(['%s' % (i * 50) for i in x],rotation=-30,fontsize='small') #ax1.set_xticklabels must used with ax1.set_xticks
ax1.set_xticklabels(labels=['x{}'.format(i for i in x)])  #设置刻度的显示文本，rotation旋转角度，fontsize字体大小
# ax1.set_xticks(x_new) #设置坐标轴刻度






# #新增区域ax2,嵌套在ax1内
# left, bottom, width, height = 0.1, 0.8, 0.2, 0.2
# # 获得绘制的句柄
# ax2 = fig.add_axes([left, bottom, width, height])
# ax2.plot(x,y, 'k-.')
# ax2.set_title('area2')
# ax2.grid(color='r', linestyle='--', linewidth=.1, alpha=1, axis='both')#linewidth 设置网格线的宽度
# # axis : 取值为‘both’， ‘x’，‘y’。就是想绘制哪个方向的网格线。alpha网格线的清晰度

plt.show()
############################################################
# x = np.array([
#     [0, 3, 4],
#     [1, 6, 4]])
# #默认参数ord=None，axis=None，keepdims=False
# print ("默认参数(矩阵整体元素平方和开根号，不保留矩阵二维特性：",np.linalg.norm(x))
# print ("矩阵整体元素平方和开根号，保留矩阵二维特性：",np.linalg.norm(x,keepdims=True))

plt.figure(3)
x_index = np.arange(5)   #柱的索引
x_data = ('A', 'B', 'C', 'D', 'E')
y1_data = (20, 35, 30, 35, 27)
y2_data = (25, 32, 34, 20, 25)
bar_width = 0.35   #定义一个数字代表每个独立柱的宽度

rects1 = plt.bar(x_index, y1_data, width=bar_width,alpha=0.4, color='b',label='legend1')            #参数：左偏移、高度、柱宽、透明度、颜色、图例
rects2 = plt.bar(x_index + bar_width, y2_data, width=bar_width,alpha=0.5,color='r',label='legend2') #参数：左偏移、高度、柱宽、透明度、颜色、图例
#关于左偏移，不用关心每根柱的中心不中心，因为只要把刻度线设置在柱的中间就可以了
plt.xticks(x_index + bar_width/2, x_data)   #x轴刻度线
plt.legend()    #显示图例
plt.tight_layout()  #自动控制图像外部边缘，此方法不能够很好的控制图像间的间隔
#############################################
# fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))     #在窗口上添加2个子图
# sigma = 1   #标准差
# mean = 0    #均值
# x=mean+sigma*np.random.randn(10000)   #正态分布随机数
# ax0.hist(x,bins=40,normed=False,histtype='bar',facecolor='yellowgreen',alpha=0.75)   #normed是否归一化，histtype直方图类型，facecolor颜色，alpha透明度
# ax1.hist(x,bins=20,normed=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8) #bins柱子的个数,cumulative是否计算累加分布，rwidth柱子宽度
plt.show()  #所有窗口运行
############################################################
def plot_0(data1, data2):
	# xdata=np.arange(0,Number_of_x)
	fig = plt.figure(figsize=(25, 10))

	axes1 = fig.add_subplot(2, 1, 1)
	axes2 = fig.add_subplot(2, 1, 2)

	# bar_width = 0.35
	axes1.plot(data1, 'b-')
	# axes1.bar(xdata,data1, color='b',width=bar_width)
	# miloc_x1 = plt.MultipleLocator(1) #
	# miloc_y1 = plt.MultipleLocator(50) #
	# axes1.xaxis.set_major_locator(miloc_x1)
	# axes1.yaxis.set_major_locator(miloc_y1)
	# axes1.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)


	axes2.plot(data2, 'r-')
	# axes2.bar(xdata,data2, color='r',width=bar_width)
	# axes2.xaxis.set_major_locator(miloc_x1)
	# axes2.yaxis.set_major_locator(miloc_y1)
	# axes2.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)



	plt.show()

def plot_1(data1, data2, Number_of_x):
	xdata=np.arange(0,Number_of_x)
	fig = plt.figure(figsize=(25, 10))

	axes1 = fig.add_subplot(2, 1, 1)
	axes2 = fig.add_subplot(2, 1, 2)

	bar_width = 0.35
	# axes1.plot(data1, 'b-')
	axes1.bar(xdata,data1/np.max(data1), color='b',width=bar_width)
	miloc_x1 = plt.MultipleLocator(1) #
	miloc_y1 = plt.MultipleLocator(0.05) #
	axes1.xaxis.set_major_locator(miloc_x1)
	axes1.yaxis.set_major_locator(miloc_y1)
	axes1.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)


	# axes2.plot(data2, 'r-')
	axes2.bar(xdata,data2/np.max(data2), color='r',width=bar_width)
	axes2.xaxis.set_major_locator(miloc_x1)
	axes2.yaxis.set_major_locator(miloc_y1)
	axes2.grid(axis='both', which='major', c='k', ls='-', lw='0.5',alpha=0.8)



	plt.show()


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


def plot_8s(ecg_datas1, ecg_datas2, label):
	fig = plt.figure(figsize=(25, 10))
	# fig = plt.figure(figsize=(35, 20))
	axes1 = fig.add_subplot(2, 1, 1)
	axes2 = fig.add_subplot(2, 1, 2)

	# axes1.plot(ecg_datas1, 'k-.')
	# axes2.plot(ecg_datas1, 'r-')

	process_axes(axes1, ecg_datas1, label)
	process_axes(axes2, ecg_datas2, label)

	plt.show()
############################################################
# def plot_8s(ecg_datas1, ecg_datas2):
# 	fig = plt.figure(figsize=(25, 10))
# 	axes1 = fig.add_subplot(2, 1, 1)
# 	axes2 = fig.add_subplot(2, 1, 2)
#
# 	process_axes(axes1, ecg_datas1)
# 	process_axes(axes2, ecg_datas2)
#
# 	plt.show()
#
# def process_axes(axes, datas):
#
# 	axes.set_xlim(0, 2500)
# 	axes.set_ylim(-2.5, 2.5)
#
# 	miloc_x1 = plt.MultipleLocator(50) #每50一个中格，共有2500/50=50个中格
# 	miloc_y1 = plt.MultipleLocator(0.5) #每0.5毫伏 一个中格
# 	axes.xaxis.set_minor_locator(miloc_x1)
# 	axes.yaxis.set_minor_locator(miloc_y1)
# 	axes.grid(axis='both', which='major', c='r', ls='-', lw='0.5')
#
# 	miloc_x2 = plt.MultipleLocator(10) #10个点一个小格，一共有50/10=5小格
# 	miloc_y2 = plt.MultipleLocator(0.1) #0.5/5 = 0.1 一个小格
# 	axes.xaxis.set_minor_locator(miloc_x2)
# 	axes.yaxis.set_minor_locator(miloc_y2)
# 	axes.grid(axis='both', which='minor', c='#ff7855', ls=':', lw='0.5')
#
# 	axes.set_xticks([50 * i for i in range(0, (1 + 50))])
# 	axes.set_yticks([(0.5 * i) for i in range(-5, 6)])
# 	axes.set_xticklabels(['%s' % (i * 50) if i % 2 == 0 else '' for i in range(1 + 50)])
# 	axes.set_yticklabels(['%smv' % (0.5 * i) for i in range(-5, 6)])
#
# 	axes.plot(datas[:], c='k', lw='0.8')
	#axes.legend()




















