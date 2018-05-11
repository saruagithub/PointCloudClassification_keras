# -*- coding:utf-8 -*-
# author:XueWang
import tensorflow as tf
import numpy as np
import sys
import os
import math
import provider
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
NUM_POINT = 1024
BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.001
DECAY_STEP = 200000
DECAY_RATE = 0.7

def dataPreHandle(train_file_idxs):
	current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs])
	current_data = current_data[:, 0:NUM_POINT, :]
	current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
	current_data = provider.rotate_point_cloud(current_data)
	current_data = provider.jitter_point_cloud(current_data)
	current_data = current_data[:, :, :, np.newaxis]
	current_label = np.squeeze(current_label)	#label
	current_label = keras.utils.to_categorical(current_label, num_classes=40)
	return current_data,current_label

train_file_idxs = np.arange(0, len(TRAIN_FILES))
np.random.shuffle(train_file_idxs)  # shuffle the file order
def generate_arrays(train_file_idxs):
	while 1:
		# for fn in range(len(TRAIN_FILES)):
			current_data,current_label = dataPreHandle(train_file_idxs[0]) #get from h5 file and handle it
			batches = current_data.shape[0]//BATCH_SIZE
			print(current_data.shape, current_label.shape,batches)
			for batch_idx in range(batches):
				start = batch_idx * BATCH_SIZE
				end = (batch_idx+1) * BATCH_SIZE
				current_data = current_data[start:end,:, :, :]
				current_label = current_label[start:end,:]
				print(current_data.shape,current_label.shape,start,end)
				yield (current_data,current_label)

heihei = generate_arrays(train_file_idxs)
#test data