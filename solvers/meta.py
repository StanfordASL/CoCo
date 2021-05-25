import pdb
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

from copy import deepcopy, copy
from datetime import datetime

sys.path.insert(1, os.environ['CoCo'])
sys.path.insert(1, os.path.join(os.environ['CoCo'], 'pytorch'))

from core import Problem, Solver
from pytorch.models import FFNet
from .coco import CoCo

class Meta(CoCo):
    """
    Constructor for meta-learning variant of CoCo
    """
    def __init__(self, system, problem, prob_features, n_evals=2, learn_lr=True):
        super().__init__(system, problem, prob_features)

        self.training_params['TRAINING_ITERATIONS'] = int(500)
        self.training_params['BATCH_SIZE'] = 32
        self.training_params['TEST_BATCH_SIZE'] = 8
        self.training_params['CHECKPOINT_AFTER'] = int(100)
        self.training_params['SAVEPOINT_AFTER'] = int(200)

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
        model_fn = 'MetaCoCo_{}_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(self.system, now)
        feas_fn = 'MetaCoCo_feas_{}_{}.pt'
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
            saved_params = list(torch.load(coco_model_fn).values())
            for ii in range(len(saved_params)-2):
                self.shared_params[ii].data.copy_(saved_params[ii])
            self.coco_last_layer[-2].data.copy_(saved_params[-2])
            self.coco_last_layer[-1].data.copy_(saved_params[-1])
            if override_model_fn:
                self.model_fn = coco_model_fn

        if os.path.exists(feas_model_fn):
            print('Loading presaved feasibility model from {}'.format(feas_model_fn))
            saved_params = torch.load(feas_model_fn)
            for ii in range(len(saved_params)):
                self.feas_last_layer[ii].data.copy_(saved_params[ii])
            self.feas_fn = feas_model_fn
        else:
            for ii in range(len(self.coco_last_layer)):
                self.feas_last_layer[ii].data.copy_(self.coco_last_layer[ii])

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
                batch_size = len(idx)
                fast_weights = [zz.clone().reshape(1, *zz.shape).expand(batch_size, *zz.shape) for zz in fast_weights]

                ff_inputs_inner = torch.from_numpy(self.features[idx]).float().to(device=self.device)
                prob_params_list = []
                for prb_idx, idx_val in enumerate(idx):
                    prob_params = {}
                    for k in params:
                        prob_params[k] = params[k][idx_val]
                    prob_params_list.append(prob_params)

                for ii_step in range(self.update_step):
                    # Skip inner loop if inaccurate
                    if len(accuracies) < self.buffer_len or np.mean(accuracies) < self.accuracy_thresh:
                        continue

                    # Use strategy classifier to identify high ranking strategies for each feature
                    feas_scores = model(ff_inputs_inner, fast_weights)

                    opt_weights = [zz.clone().reshape(1, *zz.shape).expand(batch_size, *zz.shape) for zz in self.shared_params+self.coco_last_layer]
                    class_scores = model(ff_inputs_inner, opt_weights)

                    # Predicted strategy index for each features
                    class_labels = torch.argmax(class_scores, axis=1).detach().cpu().numpy()

                    y_guesses = []
                    for prb_idx in range(len(idx)):
                        # Grab strategy indices for a particular problem
                        cl_idx = np.where(self.labels[:,0] == class_labels[prb_idx])[0][0]
                        y_guess = self.labels[cl_idx,1:]
                        y_guesses.append(np.reshape(y_guess, self.y_shape))

                    # TODO(acauligi): use multiprocessing to parallelize this
                    prob_success_flags = -np.ones(len(idx))
                    for prb_idx, idx_val in enumerate(idx):
                        try:
                            try:
                                prob_success, _, _, optvals = self.problem.solve_pinned(prob_params_list[prb_idx], y_guesses[prb_idx], solver=cp.MOSEK)
                            except:
                                prob_success, _, _, optvals = self.problem.solve_pinned(prob_params_list[prb_idx], y_guesses[prb_idx], solver=cp.GUROBI)
                        except:
                            prob_success = False

                        prob_success_flags[prb_idx] = 1. if prob_success else -1.

                        # For successful optimization problems, propagate the initial state of robot forward when applicable
                        ff_inputs_next = ff_inputs_inner.clone()
                        if prob_success_flags[prb_idx] == 1.:
                            feature_vec = ff_inputs_next[prb_idx].detach().cpu().numpy()
                            u0 = optvals[1][:,0]

                            ff_inputs_next[prb_idx] = torch.from_numpy(self.problem.propagate_features(feature_vec, u0, self.prob_features)).float().to(device=self.device)
                    prob_success_flags = torch.from_numpy(prob_success_flags).to(device=self.device)

                    # Compute hinge loss using class score from each applied strategy
                    inner_loss = torch.zeros(len(idx)).to(device=self.device)
                    for prb_idx, idx_val in enumerate(idx):
                        inner_loss[prb_idx] = feas_scores[prb_idx, class_labels[prb_idx]]

                    # If problem feasible, push scores to positive value
                    # If problem infeasible, push scores to negative value
                    inner_loss = torch.mean(torch.relu(self.margin - prob_success_flags * inner_loss))

                    # Descent step on feas_model network weights
                    grad = torch.autograd.grad(inner_loss, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - torch.square(self.update_lr_sqrt) * p[0], zip(grad, fast_weights)))

                    ff_inputs_inner = ff_inputs_next

                # Construct features for meta training loss
                ff_inputs = torch.from_numpy(self.features[idx]).float().to(device=self.device)
                labels = torch.from_numpy(self.labels[idx, 0]).long().to(device=self.device)

                # Pass inner loop weights to CoCo classifier (except last layer)
                outer_weights = [zz.clone().reshape(1, *zz.shape).expand(batch_size, *zz.shape) for zz in self.coco_last_layer]
                outer_weights = fast_weights[:-2] + outer_weights

                outputs = model(ff_inputs, outer_weights)

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

                    ff_inputs = torch.from_numpy(self.features[test_inds]).float().to(device=self.device)
                    labels = torch.from_numpy(self.labels[test_inds, 0]).long().to(device=self.device)

                    opt_weights = [zz.clone().reshape(1, *zz.shape).expand(TEST_BATCH_SIZE, *zz.shape) for zz in model.z0]
                    outputs = model(ff_inputs, opt_weights)

                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float()).detach().cpu()

                    verbose and print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                    writer.add_scalar('Loss/train', running_loss / float(self.training_params['CHECKPOINT_AFTER']) / float(BATCH_SIZE), itr)
                    writer.add_scalar('Loss/test', loss / float(TEST_BATCH_SIZE), itr)
                    writer.add_scalar('Loss/accuracy', accuracy.item(), itr)
                    running_loss = 0.

                if itr % self.training_params['SAVEPOINT_AFTER'] == 0:
                    self.save_network()

                itr += 1
            verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        self.save_network()
        verbose and print('Saved model at {}'.format(self.model_fn))
        verbose and print('Saved feas model at {}'.format(self.feas_fn))

        print('Saved model at {}'.format(self.model_fn))
        print('Saved feas model at {}'.format(self.feas_fn))

    def forward(self, prob_params, solver=cp.MOSEK):
        model = self.model

        features = self.problem.construct_features(prob_params, self.prob_features)
        inpt = Variable(torch.from_numpy(features)).float().unsqueeze(0).to(device=self.device)

        fast_weights = self.shared_params + self.feas_last_layer
        fast_weights = [zz.clone().reshape(1, *zz.shape).expand(1, *zz.shape) for zz in fast_weights]
        opt_weights = [zz.clone().reshape(1, *zz.shape).expand(1, *zz.shape) for zz in self.shared_params+self.coco_last_layer]

        inner_losses = []

        prob_successes = [] 
        for _ in range(self.update_step):
            scores = model(inpt, opt_weights).cpu().detach().numpy()[:]
            feas_scores = model(inpt, fast_weights)[0]
            torch.cuda.synchronize()
            ind_max = np.argsort(scores, axis=1)[0][-self.n_evals:][::-1]

            y_guesses = -1*np.ones((self.n_evals, self.n_y), dtype=int)
            for jj in range(self.num_train):
                # first index of training label is that strategy's idx
                label = self.labels[jj]
                if label[0] in np.unique(ind_max):
                    if label[0] in ind_max:
                        idx = np.where(ind_max==label[0])[0][0]
                        # remainder of training label is that strategy's binary pin
                        y_guesses[idx] = label[1:]

            feas_losses = []
            prob_success, cost, n_evals, optvals = False, np.Inf, self.n_evals, None
            for ii,idx in enumerate(ind_max):
                y_guess = np.reshape(y_guesses[ii], self.y_shape)
                try:
                    prob_success, cost, n_evals, optvals = self.problem.solve_pinned(prob_params, y_guess, solver=solver)
                except:
                    prob_success, cost, n_evals, optvals = False, np.Inf, 0, None

                n_evals = ii+1

                # If problem feasible, push scores to positive value
                # If problem infeasible, push scores to negative value
                sign = 1.0 if prob_success else -1.0
                feas_loss = torch.relu(self.margin - sign*feas_scores[idx])
                feas_losses.append(feas_loss)

                inpt_next = inpt.clone()
                if prob_success:
                    feature_vec = inpt_next[0].detach().cpu().numpy()
                    u0 = optvals[1][:,0]
                    inpt_next = torch.from_numpy(self.problem.propagate_features(feature_vec, u0, self.prob_features)).float().to(device=self.device)
                    break

            prob_successes += [prob_success]

            # Loss is mean over attempted solves
            inner_losses.append(torch.mean(torch.stack(feas_losses)))

        # Descent step on feas_model network weights
        inner_loss = torch.mean(torch.stack(inner_losses))
        grad = torch.autograd.grad(inner_loss, fast_weights, create_graph=True)
        fast_weights = list(map(lambda p: p[1] - torch.square(self.update_lr_sqrt) * p[0], zip(grad, fast_weights)))

        return prob_successes 
