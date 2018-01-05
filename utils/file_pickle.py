# -*- coding:utf-8 -*-
import re
import pickle


# 读入pickle的文件
def get_pickle_data(file_name):
    pkl_file = open(file_name, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    return data


# 保存pickle的文件
def save_pickle_data(file_name, data):
    output = open(file_name, 'wb')
    pickle.dump(data, output, protocol=2)
    output.close()
