import os
import glob
import random
import numpy as np
import data_info as di

import matplotlib.pyplot as plt

INDEX_LIST_TRAIN = list(range(0, 1))

YUV_PATH_ORI = 'D:\zy\\'  # path storing 12 raw YUV files
INFO_PATH = 'D:\zy\\'  # path storing Info_XX.dat files for All-Intra configuration

YUV_NAME_LIST_FULL = di.YUV_NAME_LIST_FULL
YUV_WIDTH_LIST_FULL = di.YUV_WIDTH_LIST_FULL
YUV_HEIGHT_LIST_FULL = di.YUV_HEIGHT_LIST_FULL


class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V


def get_file_size(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)


# yuv文件的总帧数
def get_num_YUV420_frame(file, width, height):
    file_bytes = get_file_size(file)
    frame_bytes = width * height * 3 // 2
    assert (file_bytes % frame_bytes == 0)
    return file_bytes // frame_bytes


def read_YUV420_a_frame(fid, width, height):
    # read a frame from a YUV420-formatted sequence
    d00 = height // 2
    d01 = width // 2
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
    U_buf = fid.read(d01 * d00)
    U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
    V_buf = fid.read(d01 * d00)
    V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
    return FrameYUV(Y, U, V)


def read_info_frame(fid, width, height):
    assert (width % 8 == 0 and height % 8 == 0)
    unit_width = 8
    num_line_in_unit = height // unit_width
    num_column_in_unit = width // unit_width

    info_buf = fid.read(num_line_in_unit * num_column_in_unit)
    info = np.reshape(np.frombuffer(info_buf, dtype=np.uint8), [num_line_in_unit, num_column_in_unit])
    return info


def write_data(fid_out, frame_Y, cu_angle_fame):
    width = np.shape(frame_Y)[1]
    height = np.shape(frame_Y)[0]
    n_line = height // 64
    n_col = width // 64
    for i_line in range(n_line):
        for i_col in range(n_col):
            buf_sample = (np.ones((4096 + 64,)) * 255).astype(np.uint8)
            patch_Y = frame_Y[i_line * 64: (i_line + 1) * 64, i_col * 64: (i_col + 1) * 64]
            buf_sample[0: 4096] = np.reshape(patch_Y, (4096,))
            patch_cu_angle = cu_angle_fame[i_line * 8: (i_line + 1) * 8, i_col * 8: (i_col + 1) * 8]
            buf_sample[4096:4096 + 64] = np.reshape(patch_cu_angle, (64,))
            fid_out.write(buf_sample)
    return n_line * n_col


def get_file_list(yuv_path_ori, info_path, yuv_name_list_full, index_list):
    yuv_name_list = [yuv_name_list_full[index] for index in index_list]
    yuv_file_list = []
    info_file_list = []

    for i in range(len(index_list)):
        yuv_file_temp = glob.glob(yuv_path_ori + yuv_name_list[i] + '.yuv')
        assert (len(yuv_file_temp) == 1)
        yuv_file_list.append(yuv_file_temp[0])

        info_file_temp = glob.glob(
            info_path + 'Qp*_' + yuv_name_list[i] + '_fn_' + '*.dat')
        assert (len(info_file_temp) == 1)
        info_file_list.append(info_file_temp[0])

    return yuv_file_list, info_file_list


def generate_data(yuv_path_ori, info_path, yuv_name_list_full,
                  yuv_width_list_full, yuv_height_list_full, index_list, save_file):
    yuv_file_list, info_file_list = get_file_list(yuv_path_ori, info_path, yuv_name_list_full, index_list)
    print(yuv_file_list)
    print(info_file_list)
    yuv_width_list = yuv_width_list_full[index_list]  # yuv 文件宽
    yuv_height_list = yuv_height_list_full[index_list]

    fid_out = open(save_file, 'wb+')
    n_sample = 0

    n_seq = len(yuv_file_list)  # 视频序列数
    for i_seq in range(n_seq):
        width = yuv_width_list[i_seq]
        height = yuv_height_list[i_seq]
        n_frame = get_num_YUV420_frame(yuv_file_list[i_seq], width, height)  # yuv 文件帧数
        fid_yuv = open(yuv_file_list[i_seq], 'rb')  # yuv 文件指针
        fid_info = open(info_file_list[i_seq], 'rb')
        for i_frame in range(n_frame):
            frame_yuv = read_YUV420_a_frame(fid_yuv, width, height)
            frame_y = frame_yuv._Y  # 一帧中的Y分量
            cu_ang_frame = read_info_frame(fid_info, width, height)
            n_sample_one_frame = write_data(fid_out, frame_y, cu_ang_frame)
            n_sample += n_sample_one_frame
            print('Seq. %d / %d, %50s : frame %d / %d, %8d samples generated.' % (
                i_seq + 1, n_seq, yuv_file_list[i_seq], i_frame + 1, n_frame, n_sample))

        fid_yuv.close()
        fid_info.close()
    fid_out.close()
    save_file_renamed = 'AI_%s_%d.dat' % (save_file, n_sample)
    os.rename(save_file, save_file_renamed)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # generate_data(YUV_PATH_ORI, INFO_PATH, YUV_NAME_LIST_FULL,
    #               YUV_WIDTH_LIST_FULL, YUV_HEIGHT_LIST_FULL, INDEX_LIST_TRAIN, 'Train')
    width = 832
    height = 480
    fid_yuv = open('D:\zy\RaceHorses_832x480_30.yuv', 'rb')
    fid_info = open('D:\zy\Qp_32_RaceHorses_832x480_30_fn_300_20230224_165601_.dat', 'rb')
    frame_yuv = read_YUV420_a_frame(fid_yuv, width, height)
    print(frame_yuv._Y)
    img = frame_yuv._Y
    print(img)
    plt.imshow(img,cmap=plt.cm.gray)
    plt.show()
