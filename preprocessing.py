from tools import preprocess
from tools.distance_compution import trajecotry_distance_list_time, trajectory_distance_combine, trajecotry_distance_list, trajectory_distance_combine_time
# import cPickle
import pickle
import numpy as np
import time


def distance_comp(coor_path):
 
    traj_coord = pickle.load(open(coor_path, 'rb'))[0][:num_traj]
    np_traj_coord = []
    np_traj_time = []
    for t in traj_coord:
        temp_coord = []
        temp_time = []
        for item in t:
            temp_coord.append([item[0], item[1]])
            temp_time.append([float(item[2]), float(0)])
        np_traj_coord.append(np.array(temp_coord))
        np_traj_time.append(np.array(temp_time))
    print(len(np_traj_coord))

    start_t = time.time()
    trajecotry_distance_list(np_traj_coord, batch_size=50, processors=20, distance_type=distance_type,
                             data_name=data_name)
    
    # trajecotry_distance_list_time(np_traj_time, batch_size=50, processors=20, distance_type=distance_type,
    #                          data_name=data_name)

    end_t = time.time()
    total = end_t - start_t
    print('Computation time is {}'.format(total))

    # trajectory_distance_combine(num_traj, batch_size=50, metric_type=distance_type, data_name=data_name)
    # trajectory_distance_combine_time(num_traj, batch_size=50, metric_type=distance_type, data_name=data_name)


if __name__ == '__main__':
    beijing_lat_range = [39.6, 40.7]
    beijing_lon_range = [115.9, 117.1]

    porto_lat_range = [40.7, 41.8]
    porto_lon_range = [-9.0, -7.9]

    # coor_path, data_name = preprocess.trajectory_feature_generation(path='./data/porto/porto',
    #                                                                 lat_range=porto_lat_range,
    #                                                                 lon_range=porto_lon_range,)
    # coor_path, data_name = preprocess.trajectory_feature_generation(path='./data/geolife/geolife',
    #                                                                 lat_range=beijing_lat_range,
    #                                                                 lon_range=beijing_lon_range,)
    # traj_coord = pickle.load(open('./features/beijing_traj_coord', 'rb'))
    # traj_index = pickle.load(open('./features/beijing_traj_index', 'rb'))
    # traj_grid = pickle.load(open('./features/beijing_traj_grid', 'rb'))

    num_traj = 9000
    data_name = 'beijing'
    distance_type = 'lcss'  # hausdorff,dtw,discret_frechet,lcss,edr,erp
    distance_comp('./features/beijing_traj_coord')

    # num_traj = 10000 
    # data_name = 'porto'
    # distance_type = 'erp'  # hausdorff,dtw,discret_frechet,lcss,edr,erp
    # distance_comp('./features/porto_traj_coord')
