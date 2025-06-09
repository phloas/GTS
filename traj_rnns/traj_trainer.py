import datetime
from torch_geometric.data import Data
import time
import os
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from tools import config
from traj_rnns.traj_model import Traj_Network
from traj_rnns.wrloss import WeightedRankingLoss
from tqdm import tqdm
import pandas as pd
import logging
import torch.autograd as autograd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU


class TrainData(Dataset):
    def __init__(self,
                 id_train_seqs, coor_train_seqs, time_train_seqs,
                 train_pos_idx, train_neg_idx, train_pos_dis, train_neg_dis,
                 train_trajs_length):
        self.id_seqs = id_train_seqs
        self.coor_seqs = coor_train_seqs
        self.time_seqs = time_train_seqs
        self.pos_idx = train_pos_idx
        self.neg_idx = train_neg_idx
        self.pos_dis = train_pos_dis
        self.neg_dis = train_neg_dis
        self.trajs_length = train_trajs_length

    def __len__(self):
        return len(self.id_seqs)

    def __getitem__(self, idx):
        return (self.id_seqs[idx], self.coor_seqs[idx], self.time_seqs[idx],
                self.pos_idx[idx], self.neg_idx[idx], self.pos_dis[idx], self.neg_dis[idx],
                self.trajs_length[idx], idx)


class ValiData(Dataset):
    def __init__(self,
                 id_vali_seqs,
                 coor_vali_seqs,
                 time_vali_seqs,
                 vali_label,
                 vali_trajs_length):
        self.id_seqs = id_vali_seqs
        self.coor_seqs = coor_vali_seqs
        self.time_seqs = time_vali_seqs
        self.label = vali_label
        self.trajs_length = vali_trajs_length
        # self.index = list(range(len(data)))

    def __len__(self):
        return len(self.id_seqs)

    def __getitem__(self, idx):
        tuple_ = (self.id_seqs[idx], self.coor_seqs[idx], self.time_seqs[idx],
                  self.label[idx],
                  self.trajs_length[idx], idx)
        # tuple_ = (self.data[idx], self.label[idx], self.index[idx])
        return tuple_


def test_spa_model(s_emb, label):

    batch_size = 100
    label_r = []
    for i in tqdm(range(0, s_emb.size(0), batch_size)):
        c_batch = s_emb[i:i + batch_size]
        diff = c_batch.unsqueeze(1) - s_emb.unsqueeze(0)
        distances_batch = torch.norm(diff, p=2, dim=2)
        label_r.append(distances_batch)
    label_r = torch.cat(label_r, dim=0)
    # label_r = label_r.cpu().numpy()
    # label_r = np.argsort(label_r)[:, 1:51]  # 升序
    label_r = torch.argsort(label_r, dim=-1, descending=False).cpu().numpy()[:, 1:51]  # 升序
    # %================
    recall = torch.zeros((s_emb.shape[0], 6))
    label = label.numpy()
    for idx, la in tqdm(enumerate(label)):
        recall[idx, 0] += len(list(set(label_r[idx, :10]).intersection(set(la[:10]))))  # HR-10
        recall[idx, 1] += len(list(set(label_r[idx, :50]).intersection(set(la[:50]))))  # HR-50
        recall[idx, 2] += len(list(set(label_r[idx, :50]).intersection(set(la[:10]))))  # R10@50
        recall[idx, 3] += len(list(set(label_r[idx, :1]).intersection(set(la[:1]))))  # R1@1
        recall[idx, 4] += len(list(set(label_r[idx, :10]).intersection(set(la[:1]))))  # R1@10
        recall[idx, 5] += len(list(set(label_r[idx, :50]).intersection(set(la[:1]))))  # R1@50

    return recall


class TrajTrainer(object):
    def __init__(self, tagset_size,
                 batch_size, sampling_num, learning_rate=config.learning_rate):

        self.target_size = tagset_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.learning_rate = learning_rate

    def data_prepare(self,
                     save_model=None,
                     train_traj_path=config.train_traj_path,
                     vali_traj_path=config.test_traj_path,
                     train_path=config.train_set_path,
                     vali_path=config.test_set_path,
                     node_path=config.node_path,
                     edge_path=config.edge_path,):
        self.save_model = save_model
        train_data = np.load(train_path, allow_pickle=True)
        self.train_pos_idx = train_data["train_y"]
        self.train_neg_idx = train_data["train_neg_y"]
        self.train_pos_dis = train_data["train_dis"]
        self.train_neg_dis = train_data["train_neg_dis"]
        self.train_dis_matrix = train_data["train_dis_matrix"]
        self.train_time_matrix = train_data["train_time_matrix"]
        # ===============================
        df_node = pd.read_csv(node_path, sep=',')
        df_dege = pd.read_csv(edge_path, sep=',')
        node_id = df_node[["lat", "lng"]].to_numpy()
        node_id[:, 0] = (node_id[:, 0]-np.mean(node_id[:, 0]))/np.std(node_id[:, 0])
        node_id[:, 1] = (node_id[:, 1]-np.mean(node_id[:, 1]))/np.std(node_id[:, 1])
        node_id = torch.tensor(node_id, dtype=torch.float32)
        edge_index = torch.LongTensor(df_dege[["s_node", "e_node"]].to_numpy()).t().contiguous()
        edge_weight = torch.tensor(1.0/df_dege["length"].to_numpy(), dtype=torch.float32)
        self.road_network = Data(x=node_id, edge_index=edge_index, edge_weight=edge_weight).cuda()
        # ===============================
        train_traj_data = np.load(train_traj_path, allow_pickle=True)
        self.coor_train_seqs = train_traj_data["coor"]
        self.time_train_seqs = train_traj_data["time"]
        self.train_trajs_length = [len(i) for i in self.coor_train_seqs]
        x, y, t = [], [], []
        for traj in self.coor_train_seqs:
            for r in traj:
                x.append(r[0])
                y.append(r[1])

        for traj in self.time_train_seqs:
            for r in traj:
                t.append(r[0])
        meanx, meany, meant, stdx, stdy, stdt = np.mean(x), np.mean(y), np.mean(t), np.std(x), np.std(y), np.std(t)
        self.time_train_seqs = [[[(r[0] - meant) / stdt]
                                 for r in t] for t in self.time_train_seqs]
        self.coor_train_seqs = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                                 for r in t] for t in self.coor_train_seqs]
        self.coor_train_seqs = np.array(self.coor_train_seqs, dtype=object)
        self.time_train_seqs = np.array(self.time_train_seqs, dtype=object)
        # ===============================
        self.id_train_seqs = train_traj_data["id"]
        # %=======================================================================================================F
        vali_data = np.load(vali_path, allow_pickle=True)
        self.vali_label = vali_data['label_idx']
        # ===============================
        vali_traj_data = np.load(vali_traj_path, allow_pickle=True)
        self.coor_vali_seqs = vali_traj_data["coor"]
        self.time_vali_seqs = vali_traj_data["time"]
        x, y, t = [], [], []
        for traj in self.coor_vali_seqs:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        for traj in self.time_vali_seqs:
            for r in traj:
                t.append(r[0])
        meanx, meany, meant, stdx, stdy, stdt = np.mean(x), np.mean(y), np.mean(t), np.std(x), np.std(y), np.std(t)
        self.time_vali_seqs = [[[(r[0] - meant) / stdt]
                                for r in t] for t in self.time_vali_seqs]
        self.coor_vali_seqs = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                                for r in t] for t in self.coor_vali_seqs]
        self.vali_trajs_length = [len(i) for i in self.coor_vali_seqs]
        self.coor_vali_seqs = np.array(self.coor_vali_seqs, dtype=object)
        self.time_vali_seqs = np.array(self.time_vali_seqs, dtype=object)
        # ===============================
        self.id_vali_seqs = vali_traj_data["id"]
        # ===============================
        logging.info(f'coe:{config.mail_pre_degree}')
# %===============================================================================================================

    def train_data_loader(self):
        def collate_fn_neg(data_tuple):

            # data, label, dis, idx_list = data_tuple
            aco = []
            pos = []
            neg = []

            aco_s = []
            pos_s = []
            neg_s = []

            aco_t = []
            pos_t = []
            neg_t = []

            aco_len = []
            pos_len = []
            neg_len = []

            pos_dis = []
            neg_dis = []
            # for idx, d in enumerate(data):
            for i, (id_seqs, coor_seqs, time_seqs, pos_idxs, neg_idxs, pos_diss, neg_diss, trajs_length, aco_idx) in (enumerate(data_tuple)):
                for j in range(len(pos_idxs)):
                    aco.append(torch.LongTensor(id_seqs))
                    aco_s.append(torch.tensor(coor_seqs, dtype=torch.float32))
                    aco_t.append(torch.tensor(time_seqs, dtype=torch.float32))
                    aco_len.append(trajs_length)

                    pos.append(torch.LongTensor(self.id_train_seqs[pos_idxs[j]]))
                    pos_s.append(torch.tensor(self.coor_train_seqs[pos_idxs[j]], dtype=torch.float32))
                    pos_t.append(torch.tensor(self.time_train_seqs[pos_idxs[j]], dtype=torch.float32))
                    pos_dis.append(np.exp(-float(pos_diss[j]*config.mail_pre_degree)))
                    pos_len.append(self.train_trajs_length[pos_idxs[j]])

                    neg_idx_random = np.random.randint(len(neg_idxs))
                    neg_idx = neg_idxs[neg_idx_random]
                    neg.append(torch.LongTensor(self.id_train_seqs[neg_idx]))
                    neg_s.append(torch.tensor(self.coor_train_seqs[neg_idx], dtype=torch.float32))
                    neg_t.append(torch.tensor(self.time_train_seqs[neg_idx], dtype=torch.float32))
                    neg_dis.append(np.exp(-float(neg_diss[neg_idx_random]*config.mail_pre_degree)))
                    neg_len.append(self.train_trajs_length[neg_idx])

                # sample_idx = torch.randint(0, int(config.datalength*config.seeds_radio), (20,))
                # while aco_idx in sample_idx:
                #     sample_idx = torch.randint(0, int(config.datalength*config.seeds_radio), (20,))
                # sample_dis = self.train_dis_matrix[aco_idx][sample_idx]
                # sample_time = self.train_time_matrix[aco_idx][sample_idx]
                # out_dis_time = config.disWeight*sample_dis+(1-config.disWeight)*sample_time
                # # all_sample_dis, sorted_idx = torch.sort(sample_dis)
                # sorted_idx = np.argsort(out_dis_time)
                # all_sample_dis = np.sort(out_dis_time)
                # all_sample_idx = sample_idx[sorted_idx]

                # train_pos_idx = all_sample_idx[:10]
                # train_pos_dis = all_sample_dis[:10]

                # train_neg_idx = all_sample_idx[10:]
                # train_neg_dis = all_sample_dis[10:]
                # for j in range(len(train_pos_idx)):
                #     aco.append(torch.LongTensor(id_seqs))
                #     aco_s.append(torch.tensor(coor_seqs, dtype=torch.float32))
                #     aco_t.append(torch.tensor(time_seqs, dtype=torch.float32))
                #     aco_len.append(trajs_length)

                #     pos.append(torch.LongTensor(self.id_train_seqs[train_pos_idx[j]]))
                #     pos_s.append(torch.tensor(self.coor_train_seqs[train_pos_idx[j]], dtype=torch.float32))
                #     pos_t.append(torch.tensor(self.time_train_seqs[train_pos_idx[j]], dtype=torch.float32))
                #     pos_dis.append(np.exp(-float(train_pos_dis[j]*config.mail_pre_degree)))
                #     pos_len.append(self.train_trajs_length[train_pos_idx[j]])

                #     neg.append(torch.LongTensor(self.id_train_seqs[train_neg_idx[j]]))
                #     neg_s.append(torch.tensor(self.coor_train_seqs[train_neg_idx[j]], dtype=torch.float32))
                #     neg_t.append(torch.tensor(self.time_train_seqs[train_neg_idx[j]], dtype=torch.float32))
                #     neg_dis.append(np.exp(-float(train_neg_dis[j]*config.mail_pre_degree)))
                #     neg_len.append(self.train_trajs_length[train_neg_idx[j]])
            aco = rnn_utils.pad_sequence(aco, batch_first=True, padding_value=0)
            aco_s = rnn_utils.pad_sequence(aco_s, batch_first=True, padding_value=0)
            aco_t = rnn_utils.pad_sequence(aco_t, batch_first=True, padding_value=0)

            pos = rnn_utils.pad_sequence(pos, batch_first=True, padding_value=0)
            pos_s = rnn_utils.pad_sequence(pos_s, batch_first=True, padding_value=0)
            pos_t = rnn_utils.pad_sequence(pos_t, batch_first=True, padding_value=0)

            neg = rnn_utils.pad_sequence(neg, batch_first=True, padding_value=0)
            neg_s = rnn_utils.pad_sequence(neg_s, batch_first=True, padding_value=0)
            neg_t = rnn_utils.pad_sequence(neg_t, batch_first=True, padding_value=0)

            return ([aco, pos, neg],
                    [aco_s, pos_s, neg_s],
                    [aco_t, pos_t, neg_t],
                    [aco_len, pos_len, neg_len],
                    [pos_dis, neg_dis])

        data_ = TrainData(self.id_train_seqs,
                          self.coor_train_seqs,
                          self.time_train_seqs,
                          self.train_pos_idx,
                          self.train_neg_idx,
                          self.train_pos_dis,
                          self.train_neg_dis,
                          self.train_trajs_length)
        dataset = DataLoader(data_, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_neg)
        return dataset

    def vali_data_loader(self):
        def collate_fn_neg(data_tuple):

            data = [torch.LongTensor(sq[0]) for sq in data_tuple]
            coor = [torch.tensor(sq[1], dtype=torch.float32) for sq in data_tuple]
            t = [torch.tensor(sq[2], dtype=torch.float32) for sq in data_tuple]
            label = [sq[3] for sq in data_tuple]
            data_length = torch.tensor([sq[4] for sq in data_tuple])
            idx = torch.tensor([sq[5] for sq in data_tuple])

            label = torch.tensor(np.array(label))
            data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
            coor = rnn_utils.pad_sequence(coor, batch_first=True, padding_value=0)
            t = rnn_utils.pad_sequence(t, batch_first=True, padding_value=0)
            return data, coor, t, label, data_length, idx

        data_ = ValiData(self.id_vali_seqs,
                         self.coor_vali_seqs,
                         self.time_vali_seqs,
                         self.vali_label,
                         self.vali_trajs_length)
        dataset = DataLoader(
            data_,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn_neg,
            drop_last=False,
        )

        return dataset
# %===============================================================================================================

    def matching_train(self,
                       save_model=False):
        self.data_prepare(save_model)
        train_data = self.train_data_loader()
        vali_data = self.vali_data_loader()

        spatial_net = Traj_Network(self.target_size,  self.batch_size, sampling_num=self.sampling_num)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                     spatial_net.parameters()), lr=config.learning_rate)
        milestones_list = [25, 50, 75, 100]
        logging.info(milestones_list)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.2)

        mse_loss_m = WeightedRankingLoss(batch_size=self.batch_size, sampling_num=self.sampling_num)

        spatial_net.cuda()
        mse_loss_m.cuda()

        best_epoch = 0
        best_hr10 = 0
        for epoch in range(config.epochs):
            spatial_net.train()
            print("Start training Epochs : {}".format(epoch))
            total_loss = 0.0
            total_pos_loss = 0.0
            total_neg_loss = 0.0
            start = time.time()
            for i, batch in tqdm(enumerate(train_data)):
                if config.use_GCN:
                    inputs_arrays = batch[0]
                else:
                    inputs_arrays = batch[1]
                time_arrays, inputs_len_arrays, target_arrays = batch[2], batch[3], batch[4]

                trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n = spatial_net(
                    inputs_arrays, time_arrays, inputs_len_arrays, self.road_network)

                positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))
                negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))

                loss = mse_loss_m.f(trajs_loss, positive_distance_target,
                                    negative_loss, negative_distance_target, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_pos_loss += mse_loss_m.trajs_mse_loss.item()
                total_neg_loss += mse_loss_m.negative_mse_loss.item()
            scheduler.step()
            logging.info(f'Used learning rate:{scheduler.get_last_lr()[0]}')
            end = time.time()
            logging.info('Epoch [{}/{}], Step [{}/{}], Epoch_Positive_Loss: {}, Epoch_Negative_Loss: {}, '
                         'Epoch_Total_Loss: {}, Time_cost: {}'.
                         format(epoch + 1, config.epochs, i + 1, len(self.coor_train_seqs) // self.batch_size,
                                total_pos_loss, total_neg_loss,
                                total_loss, end - start))

# %=======================================================================
            if epoch % 1 == 0:
                spatial_net.eval()
                with torch.no_grad():
                    start_vali = time.time()
                    if config.use_Time_encoder:
                        vali_embedding = torch.zeros((len(self.coor_vali_seqs), int(config.d*2))).cuda()
                    else:
                        vali_embedding = torch.zeros((len(self.coor_vali_seqs), int(config.d))).cuda()
                    vali_label = torch.zeros((len(self.coor_vali_seqs), 50), requires_grad=False, dtype=torch.long)

                    for i, batch in tqdm(enumerate(vali_data)):

                        if config.use_GCN:
                            aco = batch[0]
                        else:
                            aco = batch[1]
                        t, label, aco_length, idx = batch[2], batch[3], batch[4], batch[5]

                        a_out, p_out, _, _ = spatial_net.matching_forward(
                            autograd.Variable(aco, requires_grad=False).cuda(),
                            autograd.Variable(t, requires_grad=False).cuda(), aco_length, self.road_network
                        )

                        vali_embedding[idx] = a_out
                        vali_label[idx] = label
                    end_vali1 = time.time()
                    acc = test_spa_model(vali_embedding, vali_label)
                    acc = acc.mean(axis=0)  # HR-10 HR-50 R10@50 R1@1 R1@10 R1@50
                    acc[0] = acc[0] / 10.0
                    acc[1] = acc[1] / 50.0
                    acc[2] = acc[2] / 10.0
                    end_vali2 = time.time()
                    logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                        config.data_type, config.distance_type, len(self.coor_vali_seqs)))
                    logging.info('vali emb time: {:.4f}, vali acc time: {:.4f}'.format(
                        end_vali1 - start_vali, end_vali2 - start_vali))
                    logging.info(acc)
                    logging.info(" ")
                    save_modelname = self.save_model + "/epoch.pkl"
                    if not os.path.exists(self.save_model):
                        os.makedirs(self.save_model)
                    torch.save(spatial_net.state_dict(), save_modelname)

                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                        logging.info(f'best epoch: {best_epoch}')
                    if epoch - best_epoch >= 100:
                        logging.info(save_modelname)
                        break

    def matching_test(self,

                      save_model=False,
                      load_model=None):
        spatial_net = Traj_Network(self.target_size,  self.batch_size, sampling_num=self.sampling_num)
        # self.test_data_prepare(config.long_test_set_path)
        self.data_prepare(save_model)
        test_data = self.vali_data_loader()
        logging.info(f'test size: {len(self.coor_vali_seqs)}')
        if load_model != None:
            logging.info(load_model)
            m = torch.load(load_model)
            spatial_net.load_state_dict(m)
            spatial_net.cuda()

            spatial_net.eval()
            with torch.no_grad():
                start_vali = time.time()
                vali_embedding = torch.zeros((len(self.coor_vali_seqs), int(128*2)), requires_grad=False).cuda()
                vali_label = torch.zeros((len(self.coor_vali_seqs), 50), requires_grad=False, dtype=torch.long)

                for i, batch in tqdm(enumerate(test_data)):

                    if config.use_GCN:
                        aco = batch[0]
                    else:
                        aco = batch[1]
                    t, label, aco_length, idx = batch[2], batch[3], batch[4], batch[5]

                    a_out, p_out, _, _ = spatial_net.matching_forward(
                        autograd.Variable(aco, requires_grad=False).cuda(),
                        autograd.Variable(t, requires_grad=False).cuda(), aco_length, self.road_network
                    )

                    vali_embedding[idx] = a_out
                    vali_label[idx] = label
                end_vali1 = time.time()
                acc = test_spa_model(vali_embedding, vali_label)
                acc = acc.mean(axis=0)  # HR-10 HR-50 R10@50 R1@1 R1@10 R1@50
                acc[0] = acc[0] / 10.0
                acc[1] = acc[1] / 50.0
                acc[2] = acc[2] / 10.0
                end_vali2 = time.time()
                logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                    config.data_type, config.distance_type, len(self.coor_vali_seqs)))
                logging.info('vali emb time: {:.4f}, vali acc time: {:.4f}'.format(
                    end_vali1 - start_vali, end_vali2 - start_vali))
                logging.info(acc)
                logging.info(" ")
