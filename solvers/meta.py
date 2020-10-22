import pdb
import time
import random
import sys
import pickle, os
import numpy as np
import cvxpy as cp

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
from datetime import datetime

sys.path.insert(1, os.environ['MLOPT'])
sys.path.insert(1, os.path.join(os.environ['MLOPT'], 'pytorch'))

from core import Problem, Solver
from pytorch.models import FFNet, CNNet
from .mlopt_ff import MLOPT_FF

class Meta(MLOPT_FF):
    """
    Constructor for meta-learning variant of MLOPT_FF
    """
    def __init__(self, system, problem, prob_features, n_evals=2):
        super().__init__(system, problem, prob_features)

        self.update_lr = 1e-5
        self.meta_lr = 0.01
        self.n_way = 5
        self.k_spt = 1
        self.k_qry = 15
        self.task_num = 4
        self.update_step = 1
        self.update_step_test = 1
        self.margin = 10.

    def construct_strategies(self, n_features, train_data, device_id=1):
        super().construct_strategies(n_features, train_data)
        super().setup_network(device_id=device_id)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'MetaCoCoFF_{}_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(self.system, now)
        feas_fn = 'MetaCoCoFF_feas_{}_{}.pt'
        feas_fn = os.path.join(os.getcwd(), feas_fn)
        self.feas_fn = feas_fn.format(self.system, now)

        self.feas_model = deepcopy(self.model).to(device=self.device)

    def load_network(self, coco_model_fn, feas_model_fn=""):
        super().load_network(coco_model_fn)

        if os.path.exists(feas_model_fn):
            print('Loading presaved feasibility model from {}'.format(feas_model_fn))
            self.feas_model.load_state_dict(torch.load(feas_model_fn))
            self.feas_fn = feas_model_fn
        else:
          self.copy_shared_params(self.feas_model, self.model)

    def copy_shared_params(self, target, source):
        """
        Copies all network weights from source NN to target NN
        except for the last layer
        """
        ff_depth = len(target.ff_layers)-1
        last_layer_names = ['ff_layers.{}.weight'.format(ff_depth), 'ff_layers.{}.bias'.format(ff_depth)]
        source_params, target_params = source.named_parameters(), target.named_parameters()
        target_params_dict = dict(target_params)
        for name, param in source_params:
            if name in last_layer_names:
                # Ignore last layer weights
                continue
            if name in target_params_dict:
                target_params_dict[name].data.copy_(param.data)

    def copy_all_but_last(self, target, source_data):
        """
        Updates target NN parameters using list of weights in source_data except for last layer
        """
        target_params = target.named_parameters()
        target_params_dict = dict(target_params)

        idx = 0
        for ii in range(len(target.conv_layers)):
            target_params_dict['conv_layers.{}.weight'.format(ii)].data.copy_(source_data[idx].data)
            idx+=1
            target_params_dict['conv_layers.{}.bias'.format(ii)].data.copy_(source_data[idx].data)
            idx+=1

        for ii in range(len(target.ff_layers)-1):
            target_params_dict['ff_layers.{}.weight'.format(ii)].data.copy_(source_data[idx].data)
            idx+=1
            target_params_dict['ff_layers.{}.bias'.format(ii)].data.copy_(source_data[idx].data)
            idx+=1

    def copy_last(self, target, source_data):
        """
        Updates last layer NN parameters in target last layer weights in source_data
        """
        target_params = target.named_parameters()
        target_params_dict = dict(target_params)

        ff_depth = len(target.ff_layers)-1
        target_params_dict['ff_layers.{}.weight'.format(ff_depth)].data.copy_(source_data[-2].data)
        target_params_dict['ff_layers.{}.bias'.format(ff_depth)].data.copy_(source_data[-1].data)

    def train(self, train_data, summary_writer_fn):
        # grab training params
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        TRAINING_ITERATIONS = 10
        SAVEPOINT_AFTER = int(200)
        CHECKPOINT_AFTER = int(1000)

        NUM_META_PROBLEMS = np.maximum(int(np.ceil(BATCH_SIZE / self.problem.n_obs)), 10)

        model = self.model
        feas_model = self.feas_model
        writer = SummaryWriter("{}".format(summary_writer_fn))

        params = train_data[0]
        X = self.features[:self.problem.n_obs*self.num_train]
        X_cnn = np.zeros((BATCH_SIZE, 3,self.problem.H,self.problem.W))
        Y = self.labels[:self.problem.n_obs*self.num_train,0]

        training_loss = torch.nn.CrossEntropyLoss()
        meta_opt = torch.optim.Adam(model.parameters(), lr=self.meta_lr, weight_decay=0.00001)

        def solve_pinned_worker(prob_params, y_guesses, idx, return_dict):
            return_dict[idx] = False
            # Sometimes Mosek fails, so try again with Gurobi
            try:
                return_dict[idx] = self.problem.solve_pinned(prob_params[idx], y_guesses[idx], solver=cp.MOSEK)[0]
            except:
                return_dict[idx] = self.problem.solve_pinned(prob_params[idx], y_guesses[idx], solver=cp.GUROBI)[0]

        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(0,X.shape[0]-1))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii,idx in enumerate(indices):
                # fast_weights are network weights for feas_model with descent steps taken
                fast_weights = list(feas_model.parameters())
                for _ in range(self.update_step):
                    inner_loss = torch.zeros(1).to(device=self.device)

                    # Update feas_model to be using fast_weights
                    self.copy_last(feas_model, fast_weights)
                    self.copy_all_but_last(feas_model, fast_weights)

                    idx_vals = np.random.randint(0,self.num_train, NUM_META_PROBLEMS)

                    prob_params_list = []
                    y_guesses = np.zeros((NUM_META_PROBLEMS, 4*self.problem.n_obs, self.problem.N-1))

                    # Compute feature vectors for each of the indices in idx_vals
                    # Note each problem itself has self.problem.n_obs task features
                    ff_inputs_inner = torch.zeros((NUM_META_PROBLEMS*self.problem.n_obs, self.n_features)).to(device=self.device)
                    cnn_inputs_inner = torch.zeros((NUM_META_PROBLEMS*self.problem.n_obs, 3, self.problem.H, self.problem.W)).to(device=self.device)

                    for ii_meta, idx_val in enumerate(idx_vals):
                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                        prob_params_list.append(prob_params)

                        ff_inner = torch.from_numpy(self.problem.construct_features(prob_params, self.prob_features))
                        ff_inputs_inner[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)] = Variable(ff_inner.repeat(self.problem.n_obs,1)).float().to(device=self.device)

                        X_cnn_inner = np.zeros((self.problem.n_obs, 3, self.problem.H, self.problem.W))
                        for ii_obs in range(self.problem.n_obs):
                            X_cnn_inner[ii_obs] = self.problem.construct_cnn_features(prob_params, \
                                            self.prob_features, \
                                            ii_obs=ii_obs)
                        cnn_inputs_inner[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)] = Variable(torch.from_numpy(X_cnn_inner)).float().to(device=self.device)

                    class_scores = model(cnn_inputs_inner, ff_inputs_inner).detach().cpu().numpy()
                    class_labels = np.argmax(class_scores, axis=1)      # Predicted strategy index for each features of length NUM_META_PROBLEMS*self.problem.n_obs

                    feas_scores = feas_model(cnn_inputs_inner, ff_inputs_inner)

                    for ii_meta in range(NUM_META_PROBLEMS):
                        # Grab strategy indices for a particular problem
                        class_labels_prb = class_labels[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)]
                        for cl_ii, cl in enumerate(class_labels_prb):
                            cl_idx = np.where(self.labels[:,0] == cl)[0][0]
                            y_obs = self.labels[cl_idx, 1:]
                            y_guesses[ii_meta, 4*cl_ii:4*(cl_ii+1)] = np.reshape(y_obs, (4, self.problem.N-1))

                    prob_success_dict = {}
                    for ii_meta in range(NUM_META_PROBLEMS):
                        prob_success_dict[ii_meta] = False
                        try:
                            prob_success_dict[ii_meta] = self.problem.solve_pinned(prob_params_list[ii_meta], y_guesses[ii_meta], solver=cp.MOSEK)[0]
                        except:
                            prob_success_dict[ii_meta] = self.problem.solve_pinned(prob_params_list[ii_meta], y_guesses[ii_meta], solver=cp.GUROBI)[0]

                    # self.model.share_memory()
                    # manager = mp.Manager()
                    # prob_success_dict = manager.dict()
                    # processes = []
                    # for process_idx in range(NUM_META_PROBLEMS):
                    #     p = mp.Process(target=solve_pinned_worker, \
                    #       args=[prob_params_list, y_guesses, process_idx, prob_success_dict])
                    #     p.start()
                    #     processes.append(p)
                    # for p in processes:
                    #     p.join()

                    for ii_meta in range(NUM_META_PROBLEMS):
                        # Grab strategy indices and feasibility scores for a particular problem
                        class_labels_prb = class_labels[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)]
                        feas_scores_prb = feas_scores[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)]

                        feas_loss = torch.zeros(1).to(device=self.device)
                        for ii_obs in range(self.problem.n_obs):
                            feas_loss += feas_scores_prb[ii_obs, class_labels_prb[ii_obs]]

                        if prob_success_dict[ii_meta]:
                            # If problem feasible, push scores to positive value
                            inner_loss += torch.relu(self.margin - feas_loss)
                        else:
                            # If problem infeasible, push scores to negative value
                          inner_loss += torch.relu(self.margin + feas_loss)
                    inner_loss /= float(NUM_META_PROBLEMS)

                    # Descent step on feas_model network weights
                    grad = torch.autograd.grad(inner_loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                # Pass inner loop weights to MLOPT classifier (except last layer)
                self.copy_all_but_last(model, fast_weights)

                ff_inputs = Variable(torch.from_numpy(X[idx,:])).float().to(device=self.device)
                labels = Variable(torch.from_numpy(Y[idx])).long().to(device=self.device)

                # forward + backward + optimize
                X_cnn = np.zeros((len(idx), 3,self.problem.H,self.problem.W))
                for idx_ii, idx_val in enumerate(idx):
                    prob_params = {}
                    for k in params:
                        prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                    X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                cnn_inputs = Variable(torch.from_numpy(X_cnn)).float().to(device=self.device)
                outputs = model(cnn_inputs, ff_inputs)

                loss = training_loss(outputs, labels).float().to(device=self.device)
                running_loss += loss.item()

                class_guesses = torch.argmax(outputs,1)
                accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                loss.backward()
                meta_opt.step()
                meta_opt.zero_grad() # zero the parameter gradients

                # Update feas_model weights
                self.copy_last(feas_model, [fw.detach() for fw in fast_weights])
                self.copy_shared_params(feas_model, model)

                if itr % CHECKPOINT_AFTER == 0:
                    rand_idx = list(np.arange(0,X.shape[0]-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    ff_inputs = Variable(torch.from_numpy(X[test_inds,:])).float().to(device=self.device)
                    labels = Variable(torch.from_numpy(Y[test_inds])).long().to(device=self.device)

                    # forward + backward + optimize
                    X_cnn = np.zeros((len(test_inds), 3,self.problem.H,self.problem.W))
                    for idx_ii, idx_val in enumerate(test_inds):
                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                        X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                    cnn_inputs = Variable(torch.from_numpy(X_cnn)).float().to(device=self.device)
                    outputs = model(cnn_inputs, ff_inputs)

                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                    print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    torch.save(feas_model.state_dict(), self.feas_fn)
                    writer.add_scalar('Loss/train', running_loss / float(SAVEPOINT_AFTER), itr)
                    running_loss = 0.

                itr += 1
            if itr % 10 == 0:
                print(itr)
