import time
import random
import collections
import sys
import itertools
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

sys.path.insert(1, os.environ['CoCo'])
sys.path.insert(1, os.path.join(os.environ['CoCo'], 'pytorch'))

from core import Problem, Solver
from pytorch.models import FFNet, CNNet
from .coco_ff import CoCo_FF

class Meta_FF(CoCo_FF):
    """
    Constructor for meta-learning variant of CoCo_FF
    """
    def __init__(self, system, problem, prob_features, n_evals=2, learn_lr=True):
        super().__init__(system, problem, prob_features)

        self.training_params['TRAINING_ITERATIONS'] = int(250)
        self.training_params['BATCH_SIZE'] = 16
        self.training_params['TEST_BATCH_SIZE'] = 4
        self.training_params['CHECKPOINT_AFTER'] = int(50)
        self.training_params['SAVEPOINT_AFTER'] = int(100)

        self.update_lr_sqrt = Variable(1e-2*torch.ones([1]), requires_grad=learn_lr)
        self.meta_lr = 3e-4
        self.update_step = 5
        self.margin = 10.

        self.accuracy_thresh = 0.4
        self.buffer_len = 10

    def construct_strategies(self, n_features, train_data, device_id=0):
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

        self.shared_params = list(self.model.z0)[:-2]
        self.coco_last_layer = list(self.model.z0)[-2:]

        self.feas_last_layer = []
        for w in self.coco_last_layer:
            w_ = torch.zeros_like(w, requires_grad=True)
            with torch.no_grad():
                w_.data.copy_(w)
            self.feas_last_layer.append(w_)

    def save_network(self):
        torch.save(self.model.z0, self.model_fn)
        torch.save(self.feas_last_layer, self.feas_fn)

    def load_network(self, coco_model_fn, override_model_fn, feas_model_fn=""):
        if os.path.exists(coco_model_fn):
            print('Loading presaved classifier model from {}'.format(coco_model_fn))
            saved_params = list(torch.load(coco_model_fn))
            for ii in range(len(saved_params)-2):
                self.shared_params[ii].data.copy_(saved_params[ii])
            self.coco_last_layer[-2].data.copy_(saved_params[-2])
            self.coco_last_layer[-1].data.copy_(saved_params[-1])
            if override_model_fn:
                self.model_fn = coco_model_fn

        if os.path.exists(feas_model_fn):
            print('Loading presaved feasibility model from {}'.format(feas_model_fn))
            saved_params = list(torch.load(feas_model_fn))
            for ii in range(len(saved_params)):
                self.feas_last_layer[ii].data.copy_(saved_params[ii])
            self.feas_fn = feas_model_fn
        else:
            for ii in range(len(self.coco_last_layer)):
                self.feas_last_layer[ii].data.copy_(self.coco_last_layer[ii])

    def prior(self, batch_size):
        """
        Returns a list of prior task_parameters, expanded to batch_size.
        """
        return [zz.clone().reshape(1, *zz.shape).expand(
            batch_size, *zz.shape) for zz in self.z0]

    def train(self, train_data, summary_writer_fn, verbose=False):
        # grab training params
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        model = self.model
        writer = SummaryWriter("{}".format(summary_writer_fn))

        params = train_data[0]

        training_loss = torch.nn.CrossEntropyLoss()

        # Hacky way to move lr to GPU and keep it in graph
        self.update_lr_sqrt = self.update_lr_sqrt.to(device=self.device).detach().requires_grad_(True)

        all_params = self.shared_params + self.coco_last_layer + self.feas_last_layer + [self.update_lr_sqrt]
        meta_opt = torch.optim.Adam(all_params, lr=self.meta_lr, weight_decay=self.weight_decay)

        # Circular buffer to track running accuracy of classifier
        accuracies = collections.deque(maxlen=self.buffer_len)

        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0

            # Sample all data points
            rand_idx = list(np.arange(0, self.num_train-1))
            random.shuffle(rand_idx)
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii,idx in enumerate(indices):
                meta_opt.zero_grad() # zero the parameter gradients

                # fast_weights are network weights for feas_model with descent steps taken
                fast_weights = self.shared_params + self.feas_last_layer

                # Expand task params to batch size
                batch_size = len(idx)*self.problem.n_obs
                fast_weights = [zz.clone().reshape(1, *zz.shape).expand(batch_size, *zz.shape) for zz in fast_weights]

                # Note each problem itself has self.problem.n_obs task features
                ff_inputs_inner = torch.zeros((len(idx)*self.problem.n_obs, self.n_features)).to(device=self.device)
                cnn_inputs_inner = torch.zeros((len(idx)*self.problem.n_obs, 3, self.problem.H, self.problem.W)).to(device=self.device)
                prob_params_list = []
                for prb_idx, idx_val in enumerate(idx):
                    prob_params = {}
                    for k in params:
                        prob_params[k] = params[k][idx_val]
                    prob_params_list.append(prob_params)

                    prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))

                    feature_vec = torch.from_numpy(self.problem.construct_features(prob_params, self.prob_features))
                    ff_inputs_inner[prb_idx_range] = feature_vec.repeat(self.problem.n_obs,1).float().to(device=self.device)

                    X_cnn_inner = np.zeros((self.problem.n_obs, 3, self.problem.H, self.problem.W))
                    for ii_obs in range(self.problem.n_obs):
                        X_cnn_inner[ii_obs] = self.problem.construct_cnn_features(prob_params, \
                                        self.prob_features, \
                                        ii_obs=ii_obs)
                    cnn_inputs_inner[prb_idx_range] = torch.from_numpy(X_cnn_inner).float().to(device=self.device)

                for ii_step in range(self.update_step):
                    # Skip inner loop if inaccurate
                    if len(accuracies) < self.buffer_len or np.mean(accuracies) < self.accuracy_thresh:
                        continue

                    # Use strategy classifier to identify high ranking strategies for each feature
                    feas_scores = model(cnn_inputs_inner, ff_inputs_inner, fast_weights)

                    opt_weights = [zz.clone().reshape(1, *zz.shape).expand(batch_size, *zz.shape) for zz in self.shared_params+self.coco_last_layer]
                    class_scores = model(cnn_inputs_inner, ff_inputs_inner, opt_weights)

                    # Predicted strategy index for each features
                    class_labels = torch.argmax(class_scores, axis=1).detach().cpu().numpy()

                    y_guesses = -1*np.ones((len(idx), 4*self.problem.n_obs, self.problem.N-1), dtype=int)
                    for prb_idx in range(len(idx)):
                        # Grab strategy indices for a particular problem
                        prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))
                        class_labels_prb = class_labels[prb_idx_range]
                        for cl_ii, cl in enumerate(class_labels_prb):
                            cl_idx = np.where(self.labels[:,0] == cl)[0][0]
                            y_obs = self.labels[cl_idx, 1:]
                            y_guesses[prb_idx, 4*cl_ii:4*(cl_ii+1)] = np.reshape(y_obs, (4, self.problem.N-1))

                    # TODO(acauligi): use multiprocessing to parallelize this
                    prob_success_flags = np.ones(len(idx))
                    for prb_idx, idx_val in enumerate(idx):
                        prob_success = False
                        try:
                            try:
                                prob_success, _, _, optvals = self.problem.solve_pinned(prob_params_list[prb_idx], y_guesses[prb_idx], solver=cp.MOSEK)
                            except:
                                prob_success, _, _, optvals = self.problem.solve_pinned(prob_params_list[prb_idx], y_guesses[prb_idx], solver=cp.GUROBI)
                        except:
                            prob_success = False
                        prob_success_flags[prb_idx] = 1. if prob_success else -1.

                        # For successful optimization problems, propagate the initial state of robot forward when applicable
                        if prob_success_flags[prb_idx] == 1.:
                            prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))

                            feature_vec = ff_inputs_inner[self.problem.n_obs*prb_idx].detach().cpu().numpy()
                            u0 = optvals[1][:,0]

                            ff_inner = torch.from_numpy(self.problem.propagate_features(feature_vec, u0, self.prob_features))
                            ff_inputs_inner[prb_idx_range] = ff_inner.repeat(self.problem.n_obs,1).float().to(device=self.device)

                    prob_success_flags = torch.from_numpy(prob_success_flags).to(device=self.device)

                    # Compute hinge loss using class score from each applied strategy
                    inner_loss = torch.zeros(len(idx)).to(device=self.device)
                    for prb_idx, idx_val in enumerate(idx):
                        prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))
                        feas_scores_prb = feas_scores[prb_idx_range]

                        feas_loss = torch.zeros(self.problem.n_obs).to(device=self.device)
                        for ii_obs in range(self.problem.n_obs):
                            feas_loss[ii_obs] = feas_scores_prb[ii_obs, class_labels_prb[ii_obs]]
                        inner_loss[prb_idx] = torch.mean(feas_loss)

                    # If problem feasible, push scores to positive value
                    # If problem infeasible, push scores to negative value
                    inner_loss = torch.mean(torch.relu(self.margin - prob_success_flags * inner_loss))

                    # Descent step on feas_model network weights
                    grad = torch.autograd.grad(inner_loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - torch.square(self.update_lr_sqrt) * p[0], zip(grad, fast_weights)))

                # Construct features for meta training loss
                ff_inputs = np.zeros((self.problem.n_obs*len(idx), self.features.shape[1]))
                cnn_inputs = np.zeros((self.problem.n_obs*len(idx), 3,self.problem.H,self.problem.W))
                labels = np.zeros((self.problem.n_obs*len(idx)))
                for prb_idx, idx_val in enumerate(idx):
                    prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))

                    prob_params = {}
                    for k in params:
                        prob_params[k] = params[k][idx_val]
                    prob_params_list.append(prob_params)

                    ff_inputs[prb_idx_range] = self.features[self.problem.n_obs*idx_val:self.problem.n_obs*(idx_val+1), :]
                    for ii_obs in range(self.problem.n_obs):
                        cnn_inputs[self.problem.n_obs*prb_idx+ii_obs] = self.problem.construct_cnn_features(prob_params, \
                                        self.prob_features, \
                                        ii_obs=ii_obs)
                    labels[prb_idx_range] = self.labels[self.problem.n_obs*idx_val:self.problem.n_obs*(idx_val+1), 0]
                ff_inputs = torch.from_numpy(ff_inputs).float().to(device=self.device)
                cnn_inputs = torch.from_numpy(cnn_inputs).float().to(device=self.device)
                labels = torch.from_numpy(labels).long().to(device=self.device)

                # Pass inner loop weights to CoCo classifier (except last layer)
                outer_weights = [zz.clone().reshape(1, *zz.shape).expand(batch_size, *zz.shape) for zz in self.coco_last_layer]

                outer_weights = fast_weights[:-2] + outer_weights
                outputs = model(cnn_inputs, ff_inputs, outer_weights)

                loss = training_loss(outputs, labels).float().to(device=self.device)
                running_loss += loss.item()

                class_guesses = torch.argmax(outputs,1)
                accuracy = torch.mean(torch.eq(class_guesses,labels).float()).detach().cpu().numpy()
                accuracies.append(accuracy)

                loss.backward()
                meta_opt.step()

                if itr % self.training_params['CHECKPOINT_AFTER'] == 0:
                    rand_idx = list(np.arange(0, self.num_train-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]

                    ff_inputs = np.zeros((self.problem.n_obs*len(test_inds), self.features.shape[1]))
                    cnn_inputs = np.zeros((self.problem.n_obs*len(test_inds), 3,self.problem.H,self.problem.W))
                    labels = np.zeros((self.problem.n_obs*len(test_inds)))
                    for prb_idx, idx_val in enumerate(test_inds):
                        prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))

                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][idx_val]
                        prob_params_list.append(prob_params)

                        ff_inputs[prb_idx_range] = self.features[self.problem.n_obs*idx_val:self.problem.n_obs*(idx_val+1), :]
                        for ii_obs in range(self.problem.n_obs):
                            cnn_inputs[self.problem.n_obs*prb_idx+ii_obs] = self.problem.construct_cnn_features(prob_params, \
                                            self.prob_features, \
                                            ii_obs=ii_obs)
                        labels[prb_idx_range] = self.labels[self.problem.n_obs*idx_val:self.problem.n_obs*(idx_val+1), 0]
                    ff_inputs = torch.from_numpy(ff_inputs).float().to(device=self.device)
                    cnn_inputs = torch.from_numpy(cnn_inputs).float().to(device=self.device)
                    labels = torch.from_numpy(labels).long().to(device=self.device)

                    opt_weights = [zz.clone().reshape(1, *zz.shape).expand(self.problem.n_obs*TEST_BATCH_SIZE, *zz.shape) for zz in model.z0]
                    outputs = model(cnn_inputs, ff_inputs, opt_weights)

                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float()).detach().cpu().numpy()

                    writer.add_scalar('Loss/train', running_loss / float(self.training_params['CHECKPOINT_AFTER']) / float(BATCH_SIZE), itr)
                    writer.add_scalar('Loss/test', loss / float(TEST_BATCH_SIZE), itr)
                    writer.add_scalar('Loss/accuracy', accuracy.item(), itr)
                    running_loss = 0.

                if itr % self.training_params['SAVEPOINT_AFTER'] == 0:
                    self.save_network()
                torch.save(self.feas_last_layer, self.feas_fn)
                itr += 1
            verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        self.save_network()
        verbose and print('Saved model at {}'.format(self.model_fn))
        verbose and print('Saved feas model at {}'.format(self.feas_fn))

    def forward(self, prob_params, solver=cp.MOSEK, max_evals=1):
        model = self.model

        t0 = time.time()

        ff_inputs = torch.zeros((self.problem.n_obs, self.n_features)).to(device=self.device)
        cnn_inputs = torch.zeros((self.problem.n_obs, 3, self.problem.H, self.problem.W)).to(device=self.device)
        features = self.problem.construct_features(prob_params, self.prob_features)
        ff_inputs = torch.from_numpy(features).repeat((self.problem.n_obs,1)).float().to(device=self.device)

        X_cnn = np.zeros((self.problem.n_obs, 3, self.problem.H, self.problem.W))
        for ii_obs in range(self.problem.n_obs):
            X_cnn[ii_obs] = self.problem.construct_cnn_features(prob_params, \
                            self.prob_features, \
                            ii_obs=ii_obs)
        cnn_inputs = torch.from_numpy(X_cnn).float().to(device=self.device)

        fast_weights = self.shared_params + self.feas_last_layer
        fast_weights = [zz.clone().reshape(1, *zz.shape).expand(self.problem.n_obs, *zz.shape) for zz in fast_weights]
        opt_weights = [zz.clone().reshape(1, *zz.shape).expand(self.problem.n_obs, *zz.shape) for zz in self.shared_params+self.coco_last_layer]

        scores = self.model(cnn_inputs, ff_inputs, opt_weights).cpu().detach().numpy()[:]
        feas_scores = self.model(cnn_inputs, ff_inputs, fast_weights)

        torch.cuda.synchronize()

        total_time = time.time()-t0

        ind_max = np.argsort(scores, axis=1)[:,-self.n_evals:][:,::-1]

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

        prob_success, cost, n_evals, optvals = False, np.Inf, 0, None

        prob_success_flags = {}
        for ii, str_tuple in enumerate(strategy_tuples):
            y_guess = -np.ones((4*self.problem.n_obs, self.problem.N-1))
            for ii_obs in range(self.problem.n_obs):
                # rows of ind_max correspond to ii_obs, column to desired strategy
                y_obs = obs_strats[ind_max[ii_obs, str_tuple[ii_obs]]]
                y_guess[4*ii_obs:4*(ii_obs+1)] = np.reshape(y_obs, (4,self.problem.N-1))

            prob_success, cost, solve_time, optvals = self.problem.solve_pinned(prob_params, y_guess, solver=solver)

            for ii_obs in range(self.problem.n_obs):
              idx_ii = ind_max[ii_obs, str_tuple[ii_obs]]
              if (idx_ii, ii_obs) in prob_success_flags.keys():
                  # Only override
                  if prob_success_flags[(idx_ii, ii_obs)] < 0.:
                      prob_success_flags[(idx_ii, ii_obs)] = 1. if prob_success else -1.
              else:
                  prob_success_flags[(idx_ii, ii_obs)] = 1. if prob_success else -1.

              if prob_success:
                  solve_time = time.time()-t0 + total_time
                  break

        feas_losses = []
        for kk, prob_success_flag in prob_success_flags.items():
            idx_ii, ii_obs = kk

            # If problem feasible, push scores to positive value
            # If problem infeasible, push scores to negative value
            feas_loss = feas_scores[ii_obs, idx_ii]
            feas_loss = torch.relu(self.margin - prob_success_flag*feas_loss)
            feas_losses.append(feas_loss)

        # Descent step on feas_model network weights
        inner_loss = torch.mean(torch.stack(feas_losses))
        grad = torch.autograd.grad(inner_loss, fast_weights, create_graph=True)
        fast_weights = list(map(lambda p: p[1] - torch.square(self.update_lr_sqrt) * p[0], zip(grad, fast_weights)))

        return prob_success, cost, n_evals, optvals
