import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from .models import FFNet, CNNet
from copy import deepcopy

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, depth=3, neurons=128, device=0): 
        super(Meta, self).__init__()

        params = {}
        params['update_lr'] = 0.01
        params['meta_lr'] = 1e-3 
        params['n_way'] = 5 
        params['k_spt'] = 1
        params['k_qry'] = 15
        params['task_num'] = 4 
        params['update_step'] = 5
        params['update_step_test'] = 10

        self.update_lr = params['update_lr']
        self.meta_lr = params['meta_lr']
        self.n_way = params['n_way']
        self.k_spt = params['k_spt']
        self.k_qry = params['k_qry']
        self.task_num = params['task_num'] 
        self.update_step = params['update_step']
        self.update_step_test = params['update_step_test']

        self.depth = depth
        self.neurons = neurons
        self.device = device

        self.mlopt_model = CNNet(num_features, channels, ff_shape, input_size)
        self.meta_opt = optim.Adam(self.mlopt_model.parameters(), lr=self.meta_lr)

        self.feas_model = CNNet(num_features, channels, ff_shape, input_size)

    def copy_shared_params(self, source, target):
        # Copy over network weights except for last layer
        last_layer_names = ['ff_layers.{}.weight'.format(depth), 'ff_layers.{}.bias'.format(self.depth)]

        source_params, target_params = source.named_parameters(), target.named_parameters()
        target_params_dict = dict(target_params)
        for name1, param1 in source_params:
            if name1 in last_layer_names:
                continue
            if name in target_params_dict:
                target_params_dict[name1].data.copy_(param1.data)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def solve_pinned(self, mlopt_model, prob_params):
        mlopt_model.to(device=torch.device('cpu'))

        # Compute forward pass for each obstacle and save the top
        # n_eval's scoring strategies in ind_max
        ind_max = np.zeros((self.problem.n_obs, self.n_evals), dtype=int)
        for ii_obs in range(self.problem.n_obs):
            scores = None

            features = self.problem.construct_features(prob_params, self.prob_features, ii_obs=None)
            inpt = Variable(torch.from_numpy(features)).unsqueeze(0).float()

            cnn_features = np.zeros((1, 3,self.problem.H,self.problem.W))
            cnn_features[0] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=ii_obs)
            cnn_inpt = Variable(torch.from_numpy(cnn_features)).float()

            scores = mlopt_model(cnn_inpt, inpt).cpu().detach().numpy()[:].squeeze(0)

            torch.cuda.synchronize()

            # ii_obs'th row of ind_max contains top scoring indices for that obstacle
            ind_max[ii_obs] = np.argsort(scores)[-self.n_evals:][::-1]

        # Loop through strategy dictionary once
        # Save ii'th stratey in obs_strats dictionary
        obs_strats = {}
        uniq_idxs = np.unique(ind_max)

        for ii,idx in enumerate(uniq_idxs):
            for jj in range(self.labels.shape[0]):
                # first index of training label is that strategy's idx
                label = self.labels[jj]
                if label[0] == idx:
                    # remainder of training label is that strategy's binary pin
                    obs_strats[idx] = label[1:]

        # Generate Cartesian product of strategy combinations
        vv = [np.arange(0,self.n_evals) for _ in range(self.problem.n_obs)]
        strategy_tuples = list(itertools.product(*vv))

        # Sample from candidate strategy tuples based on "better" combinations
        probs_str = [1./(np.sum(st)+1.) for st in strategy_tuples]  # lower sum(st) values --> better
        probs_str = probs_str / np.sum(probs_str)
        str_idxs = np.random.choice(np.arange(0,len(strategy_tuples)), max_evals, p=probs_str)

        # Manually add top-scoring strategy tuples
        if 0 in str_idxs:
            str_idxs = np.unique(np.insert(str_idxs, 0, 0))
        else:
            str_idxs = np.insert(str_idxs, 0, 0)[:-1]
        strategy_tuples = [strategy_tuples[ii] for ii in str_idxs]

        # class_labels will denote if a feasible solution was ever found for a strategy+obstacle pair
        class_labels = {}
        for ii, str_tuple in enumerate(strategy_tuples):
            for ii_obs in range(self.problem.n_obs):
                str_idx = ind_max[ii_obs, str_tuple[ii_obs]]
                class_labels[(ii_obs,str_idx)] = False 

        for ii, str_tuple in enumerate(strategy_tuples):
            y_guess = -np.ones((4*self.problem.n_obs, self.problem.N-1))
            for ii_obs in range(self.problem.n_obs):
                # rows of ind_max correspond to ii_obs, column to desired strategy
                y_obs = obs_strats[ind_max[ii_obs, str_tuple[ii_obs]]]
                y_guess[4*ii_obs:4*(ii_obs+1)] = np.reshape(y_obs, (4,self.problem.N-1))

            prob_success = self.problem.solve_pinned(prob_params, y_guess, solver=solver)[0]

            if prob_success:
                for ii_obs in range(self.problem.n_obs):
                    str_idx = ind_max[ii_obs, str_tuple[ii_obs]]
                    class_labels[(ii_obs,str_idx)] = True

        return class_labels

    def forward(self, X_spt, Y_spt, X_qry, Y_qry):
        """

        :param X_spt:   [b, setsz, c_, h, w]
        :param Y_spt:   [b, setsz]
        :param X_qry:   [b, querysz, c_, h, w]
        :param Y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = X_spt.size()
        querysz = X_qry.size(1)

        mlopt_model_copy = deepcopy(self.mlopt_model)
        self.copy_shared_params(self.mlopt_model, self.feas_model)

        losses_q = [0 for _ in range(task_num)] 
        fast_weights = [p for p in self.feas_model.parameters()]

        for task_ii in range(self.task_num):
            loss_q = 0.

            for kk in range(1, self.update_step):
                # Compute loss on task and descent step on task parameters
                inputs = Variable(torch.from_numpy(X_spt[task_ii, idx, :])).float().to(device=self.device)
                labels = Variable(torch.from_numpy(Y_spt[task_ii, idx])).long().to(device=self.device)
    
                self.copy_shared_params(self.feas_model, mlopt_model_copy)
                class_labels = self.solve_pinned(mlopt_model, prob_params)
                for feature, prob_feasible in class_labels.iteritems():

                grad = torch.autograd.grad(loss, fast_weights) 
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))) 

            self.copy_shared_params(self.feas_model, self.mlopt_model)

            # Compute loss on meta-test set using task parameters 
            inputs = Variable(torch.from_numpy(X_qry[task_ii, idx, :])).float().to(device=self.device)
            labels = Variable(torch.from_numpy(Y_qry[task_ii, idx])).long().to(device=self.device)
            outputs = self.mlopt_model(inputs) 
            losses_q[task_ii] = training_loss(outputs, labels).float().to(device=self.device)

        # zero the parameter gradients
        self.meta_opt.zero_grad()

        loss = sum(losses_q)
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
        self.meta_opt.step()

        del mlopt_model_copy
