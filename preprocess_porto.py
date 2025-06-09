import csv
import numpy as np
# import cPickle
import pickle
from datetime import datetime
from tqdm import tqdm

porto_lon_range = [-9.0, -7.9]
porto_lat_range = [40.7, 41.8]

csvFile = open('dataset/Porto/train.csv', 'r')
reader = csv.reader(csvFile)
traj_missing = []
trajectories = []
min_lon, max_lon, min_lat, max_lat = -7.0, -10.0, 43.0, 40.0

for item in tqdm(reader):
    if (reader.line_num == 1):
        continue
    if (item[7] == 'True'):
        traj_missing.append(item[8])
    if (item[7] == 'False'):
        trajectories.append((item[8][2:-2].split('],['), item[5], item[0]))

traj_porto = []
j=0
for trajs in tqdm(trajectories):
    if (len(trajs[0]) >= 10):
        # print(trajs)
        Traj = []
        inrange = True
        tmp_min_lon = min_lon
        tmp_max_lon = max_lon
        tmp_min_lat = min_lat
        tmp_max_lat = max_lat
        i = 0
        for traj in trajs[0]:
            tr = traj.split(',')
            # print(traj)
            # print(tr)
            if (tr[0] != '' and tr[1] != ''):
                lon = float(tr[0])
                lat = float(tr[1])
                if ((lat < porto_lat_range[0]) | (lat > porto_lat_range[1]) | (lon < porto_lon_range[0]) | (lon > porto_lon_range[1])):
                    inrange = False
                if (lon < tmp_min_lon):
                    tmp_min_lon = lon
                if (lon > tmp_max_lon):
                    tmp_max_lon = lon
                if (lat < tmp_min_lat):
                    tmp_min_lat = lat
                if (lat > tmp_max_lat):
                    tmp_max_lat = lat
                traj_tup = (lon, lat)
                point_time = int(trajs[-2])+i*15
                start_dtime = datetime.fromtimestamp(point_time)
                # print(start_dtime)
                i = i+1
                Traj.append((lon, lat, point_time, start_dtime))
    if (inrange != False):
        traj_porto.append(Traj)
        min_lon = tmp_min_lon
        max_lon = tmp_max_lon
        min_lat = tmp_min_lat
        max_lat = tmp_max_lat
        j+=1
    if j>=40000:
        break

print(traj_porto[0])
print(len(traj_porto))  # 1709303
print(min_lon)
print(max_lon)
print(min_lat)
print(max_lat)

traj_w = traj_porto

with open('data/porto/porto', 'wb') as f:
    pickle.dump(traj_w, f)
