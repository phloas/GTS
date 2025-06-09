
from torch.nn import Module
from tools import config
import torch.autograd as autograd
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = torch.nn.Linear(1, 1).cuda()
        self.w = torch.nn.Linear(in_features, out_features-1).cuda()
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.f(self.w(tau))
        v2 = self.w0(tau)
        return torch.cat([v1, v2], dim=-1)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = torch.nn.Linear(1, 1).cuda()
        self.w = torch.nn.Linear(in_features, out_features-1).cuda()
        self.f = torch.cos

    def forward(self, tau):
        v1 = self.f(self.w(tau))
        v2 = self.w0(tau)
        return torch.cat([v1, v2], dim=-1)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class Co_Att(nn.Module):
    def __init__(self, dim):
        super(Co_Att, self).__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.temperature = dim ** 0.5
        self.FFN = nn.Sequential(
            nn.Linear(dim, int(dim*2)),
            nn.ReLU(),
            nn.Linear(int(dim*2), dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_t):
        # h = torch.stack([seq_s, seq_t], 2)  # [n, 2, dim]
        h = seq_t
        # print('shape of h is: ', h.shape)
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        # print('shape of attn is: ', attn.shape)
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)

        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)

        # att_s = attn_o[:, :, 0, :]
        # att_t = attn_o[:, :, 1, :]

        return attn_o


class GCN_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_model, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class SMNEncoder(Module):
    def __init__(self, hidden_size):
        super(SMNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.mlp_ele = torch.nn.Linear(2, int(hidden_size/2)).cuda()
        self.mlp_ele_t = torch.nn.Linear(1, hidden_size).cuda()
        self.st = torch.nn.Linear(int(2*hidden_size), hidden_size).cuda()

        # self.nonLeaky = torch.nn.LeakyReLU(0.1)

        self.nonLeaky = F.tanh
        self.time_embedding = Time2Vec('sin', hidden_size)
        self.gcn_model = GCN_model(int(hidden_size/2), hidden_size, int(hidden_size/2)).cuda()

        if config.use_TMN is False:
            self.mlp_ele_tmn = torch.nn.Linear(int(hidden_size/2), int(hidden_size)).cuda()

        self.seq_model_layer = 1
        self.seq_model = torch.nn.LSTM(int(hidden_size), int(hidden_size), num_layers=self.seq_model_layer)
        self.res_linear1 = torch.nn.Linear(int(hidden_size), int(hidden_size)).cuda()
        self.res_linear2 = torch.nn.Linear(int(hidden_size), int(hidden_size)).cuda()
        self.res_linear3 = torch.nn.Linear(int(hidden_size), int(hidden_size)).cuda()

        # self.mlp_ele_t = torch.nn.Linear(1, int(hidden_size)).cuda()
        # self.time_model = Co_Att(int(hidden_size))
        self.seq_model_t = torch.nn.LSTM(int(hidden_size), int(hidden_size), num_layers=self.seq_model_layer)
        self.res_linear1_t = torch.nn.Linear(int(hidden_size), int(hidden_size)).cuda()
        self.res_linear2_t = torch.nn.Linear(int(hidden_size), int(hidden_size)).cuda()
        self.res_linear3_t = torch.nn.Linear(int(hidden_size), int(hidden_size)).cuda()

    def init_hidden(self, hidden_dim, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, mini_batch_size, hidden_dim)
        return (torch.zeros(self.seq_model_layer, batch_size, hidden_dim).cuda(),
                torch.zeros(self.seq_model_layer, batch_size, hidden_dim).cuda())

    def f(self, inputs_a, inputs_b, network_data):
        input_a, input_a_t, input_len_a = inputs_a  # porto inputs:220x149x4 inputs_len:list
        input_b, input_b_t,  input_len_b = inputs_b

        # %===================
        if config.use_GCN:
            x = self.mlp_ele(network_data.x)
            graph_node_embeddings = self.gcn_model(x, network_data.edge_index, network_data.edge_weight)
            mlp_input_a = graph_node_embeddings[input_a]
            mlp_input_b = graph_node_embeddings[input_b]

        else:
            mlp_input_a = self.nonLeaky(self.mlp_ele(input_a))
            mlp_input_b = self.nonLeaky(self.mlp_ele(input_b))
        mask_a = (input_a_t[:, :, 0] != 0).unsqueeze(-2).cuda()
        mask_b = (input_b_t[:, :, 0] != 0).unsqueeze(-2).cuda()

        # %==============================================================================
        if config.use_Time_encoder:
            input_a_t = self.time_embedding(input_a_t)
            outputt_a, (hn_a, cn_a) = self.seq_model_t(input_a_t.permute(1, 0, 2))
            outputt_ca = F.sigmoid(self.res_linear1_t(outputt_a)) * self.nonLeaky(self.res_linear2_t(outputt_a))
            outputt_hata = F.sigmoid(self.res_linear3_t(outputt_a)) * self.nonLeaky(outputt_ca)
            outputt_fa = outputt_a + outputt_hata
            mask_outt_a = []
            for b, v in enumerate(input_len_a):
                mask_outt_a.append(outputt_fa[v - 1][b, :].view(1, -1))
            fa_outputt = torch.cat(mask_outt_a, dim=0)
        # %==================================================================================
        if config.use_TMN:
            scores_a_o = torch.matmul(mlp_input_a, mlp_input_b.transpose(-2, -1))
            scores_a_o = scores_a_o.masked_fill(mask_b == 0, float('-inf')).transpose(-2, -1)
            scores_a_o = scores_a_o.masked_fill(mask_a == 0, float('-inf')).transpose(-2, -1)
            scores_a = scores_a_o
            p_attn_a = F.softmax(scores_a, dim=-1)
            p_attn_a = p_attn_a.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
            p_attn_a = p_attn_a.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
            attn_ab = p_attn_a.unsqueeze(-1)
            sum_traj_b = mlp_input_b.unsqueeze(-3).mul(attn_ab).sum(dim=-2)
            cell_input_a = torch.cat((mlp_input_a, (mlp_input_a-sum_traj_b)), dim=-1)
        else:
            cell_input_a = self.mlp_ele_tmn(mlp_input_a)
        # %=====================
        if config.use_Time_encoder is False:
            input_a_t = self.mlp_ele_t(input_a_t)
            cell_input_a = self.st(torch.cat((cell_input_a, input_a_t), dim=-1))

        outputs_a, (hn_a, cn_a) = self.seq_model(cell_input_a.permute(1, 0, 2))
        outputs_ca = F.sigmoid(self.res_linear1(outputs_a)) * self.nonLeaky(self.res_linear2(outputs_a))
        outputs_hata = F.sigmoid(self.res_linear3(outputs_a)) * self.nonLeaky(outputs_ca)  # F.tanh(outputs_ca)
        outputs_fa = outputs_a + outputs_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(outputs_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)
        # ===============================
        if config.use_Time_encoder:
            out_a = torch.cat((fa_outputt, fa_outputs), dim=-1)
        else:
            out_a = fa_outputs
        # =======================================================================================
        if config.use_Time_encoder:
            input_b_t = self.time_embedding(input_b_t)
            outputt_b, (hn_b, cn_b) = self.seq_model_t(input_b_t.permute(1, 0, 2))
            outputt_cb = F.sigmoid(self.res_linear1_t(outputt_b)) * self.nonLeaky(self.res_linear2_t(outputt_b))
            outputt_hatb = F.sigmoid(self.res_linear3_t(outputt_b)) * self.nonLeaky(outputt_cb)
            outputt_fb = outputt_b + outputt_hatb
            mask_outt_b = []
            for b, v in enumerate(input_len_b):
                mask_outt_b.append(outputt_fb[v - 1][b, :].view(1, -1))
            fb_outputt = torch.cat(mask_outt_b, dim=0)
        # ============================================================================================
        if config.use_TMN:
            scores_b = scores_a_o.permute(0, 2, 1)
            p_attn_b = F.softmax(scores_b, dim=-1)
            p_attn_b = p_attn_b.masked_fill(mask_a == 0, 0.0).transpose(-2, -1)
            p_attn_b = p_attn_b.masked_fill(mask_b == 0, 0.0).transpose(-2, -1)
            attn_ba = p_attn_b.unsqueeze(-1)
            sum_traj_a = mlp_input_a.unsqueeze(-3).mul(attn_ba).sum(dim=-2)
            cell_input_b = torch.cat((mlp_input_b, (mlp_input_b - sum_traj_a)), dim=-1)
        else:
            cell_input_b = self.mlp_ele_tmn(mlp_input_b)
        # =======================
        if config.use_Time_encoder is False:
            input_b_t = self.mlp_ele_t(input_b_t)
            cell_input_b = self.st(torch.cat((cell_input_b, input_b_t), dim=-1))
        outputs_b, (hn_b, cn_b) = self.seq_model(cell_input_b.permute(1, 0, 2))
        outputs_cb = F.sigmoid(self.res_linear1(outputs_b)) * self.nonLeaky(self.res_linear2(outputs_b))
        outputs_hatb = F.sigmoid(self.res_linear3(outputs_b)) * self.nonLeaky(outputs_cb)
        outputs_fb = outputs_b + outputs_hatb
        mask_out_b = []
        for b, v in enumerate(input_len_b):
            mask_out_b.append(outputs_b[v - 1][b, :].view(1, -1))
        fb_outputs = torch.cat(mask_out_b, dim=0)
        # ========================
        if config.use_Time_encoder:
            out_b = torch.cat((fb_outputt, fb_outputs), dim=-1)
        else:
            out_b = fb_outputs
        return out_a, out_b, outputs_fa, outputs_fb  # , p_attn_a, p_attn_b


class Traj_Network(Module):
    def __init__(self, target_size,  batch_size, sampling_num):
        super(Traj_Network, self).__init__()
        self.target_size = target_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.smn = SMNEncoder(self.target_size).cuda()

    def forward(self, inputs_arrays, time_arrays, inputs_len_arrays, network_data):
        anchor_input = torch.Tensor(inputs_arrays[0]).cuda()
        anchor_input_t = torch.Tensor(time_arrays[0]).cuda()
        trajs_input = torch.Tensor(inputs_arrays[1]).cuda()
        trajs_input_t = torch.Tensor(time_arrays[1]).cuda()
        negative_input = torch.Tensor(inputs_arrays[2]).cuda()
        negative_input_t = torch.Tensor(time_arrays[2]).cuda()

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn.f(
            [anchor_input,  anchor_input_t, anchor_input_len],
            [trajs_input, trajs_input_t, trajs_input_len],
            network_data)
        trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))

        anchor_embedding, negative_embedding, outputs_an, outputs_n = self.smn.f(
            [anchor_input,  anchor_input_t, anchor_input_len],
            [negative_input, negative_input_t, negative_input_len],
            network_data)
        negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))

        return trajs_loss, negative_loss, outputs_ap, outputs_p, outputs_an, outputs_n

    def matching_forward(self, anchor_input, anchor_time, anchor_input_len, network_data):

        anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn.f(
            [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_time, anchor_input_len],
            [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_time, anchor_input_len],
            network_data)

        return anchor_embedding, trajs_embedding, outputs_ap, outputs_p
