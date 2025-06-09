import torch
import logging
import os
import logging
saveModel = False

tripleWeight = 0.99
triEpoch = -1
# Training Parameters
method_name = "matching"  # "neutraj" or "matching" or "t2s" or "t3s" or "srn"
GPU = "3"  # "0"
# device = torch.device('cuda:' + GPU) if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 0.001  # 0.005
seeds_radio = 0.2  # default:0.2
epochs = 100
batch_size = 20  # 20
sampling_num = 10
data_type = 'beijing'
distance_type = 'erp'  # hausdorff,dtw,discret_frechet,lcss,edr,erp
if distance_type == 'lcss':
    disWeight = 0.9
    tripleLoss = False
else:
    disWeight = 0.5
    tripleLoss = True
# elif distance_type == 'discret_frechet':
#     disWeight = 0.67
#     tripleLoss = True

train_set_path = os.path.join('features', data_type, distance_type, 'train_set.npz')
test_set_path = os.path.join('features', data_type, distance_type, 'test_set.npz')
node_path = os.path.join('features', data_type, data_type + '_node.csv')
delta_s = 150  # 100,125,175,200
edge_path = os.path.join('features', data_type, data_type + '_edge_'+str(delta_s)+'.csv')

train_traj_path = os.path.join('features', data_type, data_type+'_train_traj_list.npz')
test_traj_path = os.path.join('features', data_type, data_type+'_test_traj_list.npz')
if distance_type == 'dtw':
    if data_type == 'porto':
        mail_pre_degree = 1
    elif data_type == 'beijing':
        mail_pre_degree = 16
elif distance_type == 'lcss':
    mail_pre_degree = 1
elif distance_type == 'erp':
    mail_pre_degree = 1
elif distance_type == 'hausdorff':
    if data_type == 'porto':
        mail_pre_degree = 8
    elif data_type == 'beijing':
        mail_pre_degree = 8
else:
    mail_pre_degree = 1

if data_type == 'porto':
    datalength = 10000  # geolife:9000 porto:10000
    em_batch = 1000
if data_type == 'beijing':
    datalength = 9000
    em_batch = 900
test_num = 8000  # int(datalength - seeds_radio * datalength)   # geolife:7200 porto:8000

# Model Parameters
d = 128
use_GCN = True
use_Time_encoder = True
use_TMN = False


def config_to_str():
    configs = 'learning_rate = {} '.format(learning_rate) + '\n' + \
              'mail_pre_degree = {} '.format(mail_pre_degree) + '\n' + \
              'training_ratio = {} '.format(seeds_radio) + '\n' + \
              'embedding_size = {}'.format(d) + '\n' + \
              'epochs = {} '.format(epochs) + '\n' + \
              'datatype = {} '.format(data_type) + '\n' + \
              'distance_type = {}'.format(distance_type) + '\n' + \
              'batch_size = {} '.format(batch_size) + '\n' + \
              'sampling_num = {} '.format(sampling_num) + '\n' + \
              'tripleLoss = {}'.format(tripleLoss) + '\n' + \
              'tripleWeight = {}'.format(tripleWeight) + '\n' + \
              'GPU = {}'.format(GPU) + '\n' + \
              'use GCN = {}'.format(use_GCN) + '\n' + \
              'use TIME = {}'.format(use_Time_encoder) + '\n' + \
              'use TMN = {}'.format(use_TMN) + '\n' + \
              'delta_s = {}'.format(delta_s) + '\n' + \
              'disWeight = {}'.format(disWeight)
    return configs


def setup_logger(fname=None):
    if not logging.root.hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=fname,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)
