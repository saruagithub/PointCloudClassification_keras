import h5py
import os


# h5 file to pcd file
def h5_to_pcd(h5_dir, save_dir):
    with h5py.File(h5_dir, "r") as hf:
        data = hf['data'].value
        print(data.shape)
    hf.close()

    # PCD file header
    header = '# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F' \
             '\nCOUNT 1 1 1\nWIDTH 2048\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2048\nDATA ascii\n'
    for row in range(data.shape[0]):
        if row > 5:
            break
        row_data = data[row]
        # every pcd file named data(i).pcd
        pcd = open(save_dir + '/play_data_test1_' + str(row) + '.pcd', 'w+')
        pcd.write(header)

        w, h = row_data.shape
        for i in range(w):
            # every row is transferred str type, then write to pcd file
            blank = ''
            for j in range(h):
                if j == h - 1:
                    blank += str(row_data[i][j])
                else:
                    blank += str(row_data[i][j]) + ' '
            blank += '\n'
            pcd.write(blank)
        pcd.close()


if __name__ == '__main__':
    h5_dir = 'pointtest/modelnet40_ply_hdf5_2048/ply_data_test1.h5'
    save_dir = 'pointtest/uploadfiles'
    # h5_to_pcd(h5_dir, save_dir)
    os.system('pcl_viewer pointtest/uploadfiles/play_data_test1_0.pcd')
    print('ok')
