import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Module, Parameter
import torch
from tools import config
import numpy as np


class WeightMSELoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        self.weight = []
        if config.method_name == "matching":
            pos_sampling_num = config.sampling_num 
        else:
            pos_sampling_num = config.sampling_num
        for i in range(batch_size):
            # self.weight.append(np.array([0.]))
            for traj_index in range(pos_sampling_num): #10):#sampling_num):
                if config.method_name == "srn":
                    self.weight.append(np.array([1]))
                else:
                    # self.weight.append(np.array([config.sampling_num - traj_index]))
                    self.weight.append(np.array([pos_sampling_num - traj_index]))
                

        self.weight = np.array(self.weight)
        sum = np.sum(self.weight)
        self.weight = self.weight / sum
        self.weight = self.weight.astype(float)
        self.weight = Parameter(torch.Tensor(self.weight).cuda(), requires_grad=False)
        #self.weight = Parameter(torch.Tensor(self.weight), requires_grad = False)
        self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, inputs, targets, isReLU=False, isSub=False):
        if not isSub:
            div = targets - inputs.view(-1, 1)
            if isReLU:
                div = F.relu(div.view(-1, 1))
            square = torch.mul(div.view(-1, 1), div.view(-1, 1))
            weight_square = torch.mul(square.view(-1, 1), self.weight.view(-1, 1))

            loss = torch.sum(weight_square)
            #loss = torch.sum(square)
            return loss

        else:
            div = targets - inputs.view

        # loss = (inputs - targets) ** 2
        # loss_mean = loss.mean(dim=-1)
        # return loss_mean

    def triple_forward(self, pos_inputs, neg_inputs, pos_targets, neg_targets):
        wweight = []
        for i in range(20):
            # wweight.append(0.)
            for traj_index in range(10):
                wweight.append(1.)
        wweight = np.array(wweight)
        sum = np.sum(wweight)
        wweight = wweight / sum
        wweight = wweight.astype(float)
        wweight = Parameter(torch.Tensor(wweight).cuda(), requires_grad=False)
        inputs_div = pos_inputs.view(-1, 1) - neg_inputs.view(-1, 1)
        targets_div = pos_targets - neg_targets
        div = F.relu(targets_div.cuda() - inputs_div.cuda())
        weighted_div = torch.mul(div.view(-1, 1), wweight.view(-1, 1))

        loss = torch.sum(weighted_div)
        return loss


class WeightedRankingLoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)

    def f(self, p_input, p_target, n_input, n_target, epoch):
        trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).cuda(), False, False)  
        negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).cuda(), False, False)
        self.trajs_mse_loss = trajs_mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = sum([trajs_mse_loss, negative_mse_loss])
        if config.tripleLoss: 
            triLoss = self.positive_loss.triple_forward(p_input, n_input, p_target, n_target)
            loss = config.tripleWeight * triLoss + (1.0-config.tripleWeight) * loss
        return loss
    

class SpaLossFun(Module):
    def __init__(self, ):
        super(SpaLossFun, self).__init__()
        self.flag = True



    def forward(self, embedding_a, embedding_p, embedding_n, pos_dis, neg_dis):

        # pos_dis = (pos_dis*self.extra_coe)
        # neg_dis = (neg_dis*self.extra_coe)

        D_ap = pos_dis.squeeze()
        D_an = neg_dis.squeeze()

        v_ap = torch.exp(-(torch.norm(embedding_a-embedding_p, p=2, dim=-1)))
        v_an = torch.exp(-(torch.norm(embedding_a-embedding_n, p=2, dim=-1)))
        loss_entire_ap = (D_ap - v_ap) ** 2
        loss_entire_an = (D_an - v_an) ** 2
        loss = loss_entire_ap + loss_entire_an + (D_ap > D_an)*(F.relu(v_an - v_ap)) ** 2
        loss_mean = loss.mean(dim=-1)
        return loss_mean
