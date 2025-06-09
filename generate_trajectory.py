import pickle
import numpy as np
import os.path as osp
from tqdm import tqdm
from tools import config
import torch
import torch.nn as nn
import datetime


def get_label(input_dis_matrix, input_time_matrix, count):
    label = []
    for i in tqdm(range(len(input_dis_matrix))):  # (5000,5000)
        input_r = np.array(input_dis_matrix[i])
        input_t = np.array(input_time_matrix[i])
        # input_r = input_r[np.where(input_r != -1)[0]]
        out = config.disWeight*input_r+(1-config.disWeight)*input_t
        # out = input_r
        idx = np.argsort(out)
        label.append(idx[1:count+1])
    return np.array(label)


def get_train_label(input_dis_matrix, input_time_matrix, count):
    label = []
    neg_label = []

    label_dis = []
    neg_label_dis = []
    for i in tqdm(range(len(input_dis_matrix))):
        input_r = np.array(input_dis_matrix[i])
        input_t = np.array(input_time_matrix[i])
        out = config.disWeight*input_r+(1-config.disWeight)*input_t
        idx = np.argsort(out)
        # label.append(idx[1])
        val_r = input_r[idx]
        val_t = input_t[idx]
        val = config.disWeight*val_r+(1-config.disWeight)*val_t
        label.append(idx[1:count+1])
        label_dis.append(val[1:count+1])
        neg_label.append(idx[count+1:])
        neg_label_dis.append(val[count+1:])

    label = np.array(label, dtype=object)
    neg_label = np.array(neg_label, dtype=object)
    label_dis = np.array(label_dis, dtype=object)
    neg_label_dis = np.array(neg_label_dis, dtype=object)
    return label, neg_label, label_dis, neg_label_dis


train_size = int(config.datalength*config.seeds_radio)
test_size = int(config.datalength)

all_list_int = np.array(pickle.load(
    open(osp.join('features', config.data_type+'_traj_coord'), 'rb'))[0], dtype=object)[:config.datalength]

all_id_list_int = np.array(pickle.load(
    open(osp.join('features', config.data_type, config.data_type+'_traj_id'), 'rb')), dtype=object)[:config.datalength]

print(config.data_type)
print(config.distance_type)
beijing_lat_range = [39.6, 40.7]
beijing_lon_range = [115.9, 117.1]

porto_lat_range = [40.7, 41.8]
porto_lon_range = [-9.0, -7.9]

all_coor_list_int = []
all_time_list_int = []
for trajs in all_list_int:
    tmp_coor = []
    tmp_time = []
    for lat, lng, timeslot, times in trajs:
        tmp_coor.append([lat, lng])
        tmp_time.append([timeslot, times])
    all_coor_list_int.append(tmp_coor)
    all_time_list_int.append(tmp_time)
all_coor_list_int = np.array(all_coor_list_int, dtype=object)
all_time_list_int = np.array(all_time_list_int, dtype=object)


train_coor_list = all_coor_list_int[0:train_size]
test_coor_list = all_coor_list_int[train_size:test_size]

train_time_list = all_time_list_int[0:train_size]
test_time_list = all_time_list_int[train_size:test_size]

train_id_list = all_id_list_int[0:train_size]
test_id_list = all_id_list_int[train_size:test_size]

dis_matrix = np.load(osp.join('ground_truth', config.data_type,
                              config.distance_type, config.distance_type + '_spatial_distance.npy'))
time_matrix = np.load(osp.join('ground_truth', config.data_type,
                               config.distance_type, config.distance_type + '_temporal_distance.npy'))

np.fill_diagonal(dis_matrix, 0)
np.fill_diagonal(time_matrix, 0)
dis_matrix = dis_matrix/np.max(dis_matrix)
time_matrix = time_matrix/np.max(time_matrix)
# dis_matrix = (dis_matrix-np.mean(dis_matrix)) / np.std(dis_matrix)
# time_matrix = (time_matrix - np.mean(time_matrix)) / np.std(time_matrix)
train_dis_matrix = dis_matrix[0:train_size, 0:train_size]
test_dis_matrix = dis_matrix[train_size:test_size, train_size:test_size]

train_time_matrix = time_matrix[0:train_size, 0:train_size]
test_time_matrix = time_matrix[train_size:test_size, train_size:test_size]

train_y, train_neg_y, train_dis, train_neg_dis = get_train_label(train_dis_matrix, train_time_matrix, 10)
test_y = get_label(test_dis_matrix, test_time_matrix, 50)  # 29894

np.savez(config.train_traj_path,
         coor=train_coor_list,
         id=train_id_list,
         time=train_time_list)

np.savez(config.train_set_path,
         train_y=train_y,
         train_neg_y=train_neg_y,
         train_dis=train_dis,
         train_neg_dis=train_neg_dis,
         train_dis_matrix=train_dis_matrix,
         train_time_matrix=train_time_matrix)

np.savez(config.test_traj_path,
         coor=test_coor_list,
         id=test_id_list,
         time=test_time_list)
np.savez(config.test_set_path,
         label_idx=test_y)
