# -*- coding:utf-8 -*-
# author:XueWang
# from __future__ import print_function
import numpy as np
import pcl
from pointtest import provider
import random

current_data, current_label = provider.loadDataFile('pointtest/modelnet40_ply_hdf5_2048/ply_data_train1.h5')
filename = 'ply_data_train1.h5'

current_data_one = current_data[0]
f_prefix = filename.split('.')[0]
output_filename = '{prefix}.pcd'.format(prefix=f_prefix)
output = open(output_filename,"w+")

list = ['# .PCD v.5 - Point Cloud Data file format\n','VERSION .5\n','FIELDS x y z\n','SIZE 4 4 4\n','TYPE F F F\n','COUNT 1 1 1\n']

output.writelines(list)
output.write('WIDTH ') #注意后边有空格
output.write(str(2048))
output.write('\nHEIGHT')
output.write(str(1))  #强制类型转换，文件的输入只能是str格式
output.write('\nPOINTS ')
output.write(str(2048))
output.write('\nDATA ascii\n')
file1 = open(filename,"r")
all = file1.read()
output.write(all)
output.close()
file1.close()


