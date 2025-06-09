import pandas as pd
import pickle
import numpy as np
import os.path as osp
from tqdm import tqdm
from tools import config
import csv
import math


beijing_lat_range = [39.6, 40.7]
beijing_lon_range = [115.9, 117.1]

porto_lat_range = [40.7, 41.8]
porto_lon_range = [-9.0, -7.9]


def extract_nodes_edges(traj_grid, traj_coor):
    traj_id = []
    grid2id = {}
    coor2id = {}
    node_id = 0
    for grids, coors in zip(traj_grid, traj_coor):
        tmp_traj = []
        for grid, coor in zip(grids, coors):
            if tuple(grid) not in grid2id:
                grid2id[tuple(grid)] = node_id
                coor2id[tuple([coor[0], coor[1]])] = node_id
                node_id += 1
            tmp_traj.append(grid2id[tuple(grid)])
        traj_id.append(tmp_traj)
    # write_nodes_to_csv(coor2id)
    # with open('./features/{}/{}_traj_id'.format(config.data_type, config.data_type), 'wb') as f:
    #     pickle.dump(traj_id, f)
    edges_100 = []
    edges_125 = []
    edges_175 = []
    edges_200 = []
    for point, point_id in tqdm(coor2id.items()):
        for other_point, other_point_id in coor2id.items():
            distance = haversine(point[0], point[1], other_point[0], other_point[1])
            if distance <= 100.0 and distance != 0.0:
                edges_100.append([point_id,  other_point_id, distance])
            if distance <= 125.0 and distance != 0.0:
                edges_125.append([point_id,  other_point_id, distance])
            if distance <= 175.0 and distance != 0.0:
                edges_175.append([point_id,  other_point_id, distance])
            if distance <= 200.0 and distance != 0.0:
                edges_200.append([point_id,  other_point_id, distance])
    write_edges_to_csv_100(edges_100)
    write_edges_to_csv_125(edges_125)
    write_edges_to_csv_175(edges_175)
    write_edges_to_csv_200(edges_200)


def write_edges_to_csv_100(edges, file_name=osp.join('features', config.data_type, config.data_type + '_edge_100.csv')):

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s_node', 'e_node', 'length'])  # 写入表头
        writer.writerows(edges)


def write_edges_to_csv_125(edges, file_name=osp.join('features', config.data_type, config.data_type + '_edge_125.csv')):

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s_node', 'e_node', 'length'])  # 写入表头
        writer.writerows(edges)
def write_edges_to_csv_150(edges, file_name=osp.join('features', config.data_type, config.data_type + '_edge.csv')):

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s_node', 'e_node', 'length'])  # 写入表头
        writer.writerows(edges)

def write_edges_to_csv_175(edges, file_name=osp.join('features', config.data_type, config.data_type + '_edge_175.csv')):

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s_node', 'e_node', 'length'])  # 写入表头
        writer.writerows(edges)


def write_edges_to_csv_200(edges, file_name=osp.join('features', config.data_type, config.data_type + '_edge_200.csv')):

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s_node', 'e_node', 'length'])  # 写入表头
        writer.writerows(edges)


def haversine(lat1, lng1, lat2, lng2):
    R = 6371*1000  # 地球半径，单位为米
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def extract_edges(trajectory, nodes_dict):

    edges = set()  # 使用集合来去重边
    for i in range(1, len(trajectory)):
        lat1, lng1 = trajectory[i - 1]
        lat2, lng2 = trajectory[i]

        # 使用节点字典将经纬度转换为节点 ID
        s_node = nodes_dict.get((lat1, lng1))
        e_node = nodes_dict.get((lat2, lng2))

        if s_node is not None and e_node is not None:
            length = haversine(lat1, lng1, lat2, lng2)  # 计算两点之间的距离
            edges.add(tuple(sorted([s_node, e_node, length])))  # 使用 sorted 保证边的顺序一致性

    # 将边转换为列表形式，并添加计数
    unique_edges = []
    edge_count = {}
    for s_node, e_node, length in edges:
        # 统计每条边的出现次数
        edge_key = (s_node, e_node, length)
        edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
    for (s_node, e_node, length), count in edge_count.items():
        unique_edges.append([s_node, e_node, length, count])

    return unique_edges


# all_coor_list_int = np.array(pickle.load(
#     open(osp.join('features', config.data_type+'_traj_coord'), 'rb'))[0], dtype=object)[0:config.datalength]
# all_grid_list_int = np.array(pickle.load(
#     open(osp.join('features', config.data_type+'_traj_grid'), 'rb'))[0], dtype=object)[0:config.datalength]
# extract_nodes_edges(all_grid_list_int, all_coor_list_int)
df_edge = pd.read_csv(config.edge_path, sep=',')
df_edge_filtered = df_edge[df_edge['length'] <= 150]
df_edge_filtered.to_csv(osp.join('./features', config.data_type, config.data_type + '_edge.csv'), index=False)
print(1)
