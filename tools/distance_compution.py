# import cPickle
import traj_dist.distance as tdist
import os
import numpy as np
import multiprocessing
import time
import pickle


def trajectory_distance(traj_feature_map, traj_keys,  distance_type="hausdorff", batch_size=50, processors=30):
    # traj_keys= traj_feature_map.keys()
    trajs = []
    for k in traj_keys:
        traj = []
        for record in traj_feature_map[k]:
            traj.append([record[1], record[2]])
        trajs.append(np.array(traj))

    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs)):
        if (i != 0) & (i % batch_size == 0):
            print(batch_size*batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
                                                         'geolife'))
            batch_number += 1
    pool.close()
    pool.join()


def trajecotry_distance_list(trajs, distance_type="hausdorff", batch_size=50, processors=30, data_name='porto'):
    print(data_name)
    print(distance_type)
    pool = multiprocessing.Pool(processes=processors)
    for i in range(len(trajs)+1):
        if (i != 0) & (i % batch_size == 0):
            print(i-batch_size, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[i-batch_size:i], trajs, distance_type,
                                                         data_name))
    pool.close()
    pool.join()

    # for i in range(len(trajs)+1):
    #     if (i != 0) & (i % batch_size == 0):
    #         print(i-batch_size, i)
    #         trajectory_distance_batch(i, trajs[i-batch_size:i], trajs, distance_type, data_name)


def trajectory_distance_batch(i, batch_trjs, trjs, metric_type="hausdorff", data_name='porto'):
    if metric_type == 'lcss':
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)  # eps=0.003
        tmp_matrix = 1.0 - trs_matrix
        len_a = len(batch_trjs)
        len_b = len(trjs)
        min_len_matrix = np.ones((len_a, len_b))
        sum_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                min_len_matrix[ii][jj] = min(len(batch_trjs[ii]), len(trjs[jj]))
                sum_len_matrix[ii][jj] = len(batch_trjs[ii]) + len(trjs[jj])
        tmp_trs_matrix = tmp_matrix * min_len_matrix
        trs_matrix = sum_len_matrix - 2.0 * tmp_trs_matrix
    elif metric_type == 'edr':
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)  # eps=0.003
        len_a = len(batch_trjs)
        len_b = len(trjs)
        max_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                max_len_matrix[ii][jj] = max(len(batch_trjs[ii]), len(trjs[jj]))
        trs_matrix = trs_matrix * max_len_matrix
    elif metric_type == 'erp':
        aa = np.zeros(2, dtype=float)
        aa[0] = 40.0  # (39.0)geolife:39.6  porto:40.7
        aa[1] = -10.0  # (115.0)geolife:115.9 porto:-9.0
        # print(aa)
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, g=aa)
    else:
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)

    trs_matrix = np.array(trs_matrix)
    p = './ground_truth/{}/{}/{}_spatial_batch/'.format(data_name, str(metric_type), str(metric_type))
    if not os.path.exists(p):
        os.makedirs(p)
    np.save('./ground_truth/{}/{}/{}_spatial_batch/{}_spatial_distance.npy'.format(data_name,
            str(metric_type), str(metric_type), str(i)), trs_matrix)
    print('complete: ' + str(i))


def trajectory_distance_combine(trajs_len, batch_size, metric_type, data_name):
    res = []
    for i in range(trajs_len + 1):
        if i != 0 and i % batch_size == 0:
            res.append(np.load('./ground_truth/{}/{}/{}_spatial_batch/{}_spatial_distance.npy'.format(data_name,
                                                                                                      str(metric_type),
                                                                                                      str(metric_type),
                                                                                                      str(i))))
    res = np.concatenate(res, axis=0)
    np.save('./ground_truth/{}/{}/{}_spatial_distance.npy'.format(data_name, str(metric_type), str(metric_type)), res)
    print('success merge similarity ground_truth')


def trajecotry_distance_list_time(trajs, distance_type="hausdorff", batch_size=50, processors=30, data_name='porto'):
    print(data_name)
    print(distance_type)
    pool = multiprocessing.Pool(processes=processors)
    for i in range(len(trajs)+1):
        if (i != 0) & (i % batch_size == 0):
            print(i-batch_size, i)
            pool.apply_async(trajectory_distance_batch_time, (i, trajs[i-batch_size:i], trajs, distance_type,
                                                              data_name))
    pool.close()
    pool.join()


def trajectory_distance_batch_time(i, batch_trjs, trjs, metric_type="hausdorff", data_name='porto'):
    if metric_type == 'lcss':
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)  # eps=0.003
        tmp_matrix = 1.0 - trs_matrix
        len_a = len(batch_trjs)
        len_b = len(trjs)
        min_len_matrix = np.ones((len_a, len_b))
        sum_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                min_len_matrix[ii][jj] = min(len(batch_trjs[ii]), len(trjs[jj]))
                sum_len_matrix[ii][jj] = len(batch_trjs[ii]) + len(trjs[jj])
        tmp_trs_matrix = tmp_matrix * min_len_matrix
        trs_matrix = sum_len_matrix - 2.0 * tmp_trs_matrix
    elif metric_type == 'edr':
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)  # eps=0.003
        len_a = len(batch_trjs)
        len_b = len(trjs)
        max_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                max_len_matrix[ii][jj] = max(len(batch_trjs[ii]), len(trjs[jj]))
        trs_matrix = trs_matrix * max_len_matrix
    elif metric_type == 'erp':
        aa = np.zeros(2, dtype=float)
        aa[0] = 1000000000  # (39.0)geolife:39.6  porto:40.7
        aa[1] = 0  # (115.0)geolife:115.9 porto:-9.0
        # print(aa)
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, g=aa)
    else:
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)

    trs_matrix = np.array(trs_matrix)
    p = './ground_truth/{}/{}/{}_temporal_batch/'.format(data_name, str(metric_type), str(metric_type))
    if not os.path.exists(p):
        os.makedirs(p)
    np.save('./ground_truth/{}/{}/{}_temporal_batch/{}_temporal_distance.npy'.format(data_name,
            str(metric_type), str(metric_type), str(i)), trs_matrix)
    print('complete: ' + str(i))


def trajectory_distance_combine_time(trajs_len, batch_size, metric_type, data_name):
    res = []
    for i in range(trajs_len + 1):
        if i != 0 and i % batch_size == 0:
            res.append(np.load('./ground_truth/{}/{}/{}_temporal_batch/{}_temporal_distance.npy'.format(data_name,
                                                                                                        str(metric_type),
                                                                                                        str(metric_type),
                                                                                                        str(i))))
    res = np.concatenate(res, axis=0)
    np.save('./ground_truth/{}/{}/{}_temporal_distance.npy'.format(data_name, str(metric_type), str(metric_type)), res)
    print('success merge similarity ground_truth')
