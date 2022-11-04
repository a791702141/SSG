from scipy.io import loadmat
import numpy as np
import sys
import pickle
from PIL import Image

base_dir = '/dfs/data/data/office_home_dg/' # dir

def load_list(base_dir, file_name):
    f = open(base_dir + file_name, "r")
    img_list = []
    label_list = []
    for lines in f.readlines():
        line_now = lines[:-1].split(" ")
        if len(line_now) >=2 :
            img_list.append(line_now[0])
            label_list.append(int(line_now[1]))

    return np.array(img_list), np.array(label_list)

def load_data_officehome(name, is_target):

    if is_target == False:
        train_list, train_label = load_list(base_dir, name + '_train.txt')
    else:
        train_list, train_label = load_list(base_dir, name + '_train.txt')
    test_list, test_label = load_list(base_dir, name + '_test.txt')


    print(name + ' train y shape->',  train_label.shape)
    print(name + ' test y shape->', test_label.shape)
    
    return train_list, train_label, test_list, test_label
