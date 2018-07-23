# -*- coding:utf-8 -*-
# author:XueWang
import h5py


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    return data, label


def write_5(h5_filename, data, label, from_index, to_index):
    count = to_index - from_index
    h5 = h5py.File(h5_filename, 'w')
    h5.create_dataset(name='data', shape=(count, 2048, 3), dtype=float, data=data[from_index:to_index])
    h5.create_dataset(name='label', shape=(count, 1), dtype=float, data=label[from_index:to_index])
    h5.close()


if __name__ == '__main__':
    test_data, test_label = load_h5('pointtest/modelnet40_ply_hdf5_2048/ply_data_test0.h5')
    print(test_data.shape, test_label.shape)
    # split the ply h5 file
    write_5('test0_0_50.h5', test_data, test_label,0,50)

