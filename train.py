from traj_rnns.traj_trainer import TrajTrainer
from tools import config
import os
from tools.config import setup_logger, zipdir
import datetime
import os.path as osp
import os
import zipfile
import pathlib
import logging

if __name__ == '__main__':

    

    trajrnn = TrajTrainer(tagset_size=config.d, batch_size=config.batch_size,
                          sampling_num=config.sampling_num)
    # %====================================================================
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_logger(osp.join('log', config.data_type, config.distance_type + '_' + str(current_time) + '_train.log'))
    save_folder = osp.join('saved_models', config.data_type, config.distance_type + '_' + str(current_time))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    zipf = zipfile.ZipFile(os.path.join(save_folder, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    logging.info('Method name: ' + config.method_name)
    logging.info(config.config_to_str())

    trajrnn.matching_train(save_model=save_folder)
    
    # current_time = '20241207_091002'
    # load_model_name = osp.join('saved_models', config.data_type, config.distance_type
    #                            + '_' + current_time, 'epoch_80.pkl')
    # setup_logger(osp.join('log', config.data_type,
    #              config.distance_type + '_' + str(current_time) + '_test.log'))
    # trajrnn.matching_test(load_model=load_model_name)
# gcn
# 时间模块
# 时间attn
