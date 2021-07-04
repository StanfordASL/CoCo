import os
import cvxpy as cp
import pickle
import numpy as np
import pdb
import time
import random
import sys
import itertools
import torch

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sigmoid
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.insert(1, os.environ['CoCo'])
sys.path.insert(1, os.path.join(os.environ['CoCo'], 'pytorch'))

from core import Problem, Solver
from pytorch.models import FFNet, CNNet

class CoCo_FF(Solver):
    def __init__(self, system, problem, prob_features, n_evals=2):
        """Constructor for CoCo FF class.

        Args:
            system: string for system name e.g. 'cartpole'
            problem: Problem object for system
            prob_features: list of features to be used
            n_evals: number of strategies attempted to be solved
        """
        super().__init__()
        self.system = system
        self.problem = problem
        self.prob_features = prob_features
        self.n_evals = n_evals

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None

        self.lr = 3e-4
        self.weight_decay = 0.00001

        # training parameters
        self.training_params = {}
        self.training_params['TRAINING_ITERATIONS'] = int(1500)
        self.training_params['BATCH_SIZE'] = 64
        self.training_params['CHECKPOINT_AFTER'] = int(1000)
        self.training_params['SAVEPOINT_AFTER'] = int(3000)
        self.training_params['TEST_BATCH_SIZE'] = 32

    def construct_strategies(self, n_features, train_data, test_data=None):
        """ Reads in training data and constructs strategy dictonary
            TODO(acauligi): be able to compute strategies for test-set too
        """
        self.n_features = n_features
        self.strategy_dict = {}
        self.binary_dict = {}

        p_train = train_data[0]
        obs_train = p_train['obstacles']
        x_train = train_data[1]
        y_train = train_data[-3]
        for k in p_train.keys():
            self.num_train = len(p_train[k])

        T = int(x_train.shape[1] / (2*self.problem.n))

        num_probs = self.num_train
        params = p_train
        self.Y = y_train

        self.n_y = int(self.Y[0].size / self.problem.n_obs)
        self.y_shape = self.Y[0].shape
        self.features = np.zeros((self.problem.n_obs*num_probs, self.n_features))
        self.cnn_features = None
        self.cnn_features_idx = None
        if "obstacles_map" in self.prob_features:
            self.cnn_features_idx = np.zeros((self.problem.n_obs*num_probs, 2), dtype=int)
        self.labels = np.zeros((self.problem.n_obs*num_probs, 1+self.n_y), dtype=int)
        self.n_strategies = 0

        for ii in range(num_probs):
            for jj in range(T):
                x_sol = x_train[ii, 2*self.problem.n*jj:2*self.problem.n*(jj+1), :]
                obs_strats = self.problem.which_M(x_sol, obs_train[ii])

                prob_params = {}
                for k in params:
                    prob_params[k] = params[k][ii]

                for ii_obs in range(self.problem.n_obs):
                    # TODO(acauligi): check if transpose necessary with new pickle save format for Y
                    y_true = np.reshape(self.Y[ii, 4*ii_obs:4*(ii_obs+1),:], (self.n_y))
                    obs_strat = tuple(obs_strats[ii_obs])

                    if obs_strat not in self.strategy_dict.keys():
                        self.strategy_dict[obs_strat] = np.hstack((self.n_strategies, y_true))
                        self.binary_dict[self.n_strategies] = y_true
                        self.n_strategies += 1

                    self.labels[ii*self.problem.n_obs+ii_obs] = self.strategy_dict[obs_strat]

                    self.features[ii*self.problem.n_obs+ii_obs] = self.problem.construct_features(prob_params, self.prob_features, ii_obs=ii_obs if 'obstacles' in self.prob_features else None)
                    if "obstacles_map" in self.prob_features:
                        self.cnn_features_idx[ii*self.problem.n_obs+ii_obs] = np.array([ii,ii_obs], dtype=int)

    def setup_network(self, depth=3, neurons=128, device_id=0):
        if device_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(device_id))

        ff_shape = []
        for ii in range(depth):
            ff_shape.append(neurons)

        ff_shape.append(self.n_strategies)
        if "obstacles_map" not in self.prob_features:
            ff_shape.insert(0, self.n_features)
            self.model = FFNet(ff_shape, activation='relu', cond_type='all_weights').to(device=self.device)
        else:
            ker = 2
            strd = 2
            pd = 0
            channels = [3, 16, 16, 16]
            H, W = 32, 32
            input_size = (H,W)

            self.model = CNNet(self.n_features, channels, ff_shape, input_size, \
                  kernel=ker, stride=strd, padding=pd, \
                  conv_activation='relu', ff_activation='relu', cond_type='mixed_weights').to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'CoCoFF_{}_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(self.system, now)

    def load_network(self, fn_classifier_model):
        if os.path.exists(fn_classifier_model):
            print('Loading presaved classifier model from {}'.format(fn_classifier_model))
            saved_params = torch.load(fn_classifier_model)
            for ii in range(len(self.model.z0)):
                self.model.z0[ii].data.copy_(saved_params[ii])
            self.model_fn = fn_classifier_model

    def train(self, train_data, summary_writer_fn, verbose=True):
        # grab training params
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        model = self.model
        model.to(device=self.device)
        writer = SummaryWriter("{}".format(summary_writer_fn))

        X = self.features
        X_cnn = None
        if 'obstacles_map' in self.prob_features:
            # X_cnn = self.cnn_features[:self.problem.n_obs*self.num_train]
            # TODO(acauligi)
            params = train_data[0]
            X_cnn = np.zeros((BATCH_SIZE, 3,self.problem.H,self.problem.W))
        Y = self.labels[:self.problem.n_obs*self.num_train,0]

        training_loss = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(model.z0, lr=self.lr, weight_decay=self.weight_decay)

        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(0, X.shape[0]-1))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii,idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                ff_inputs = Variable(torch.from_numpy(X[idx,:])).float().to(device=self.device)
                labels = Variable(torch.from_numpy(Y[idx])).long().to(device=self.device)

                # forward + backward + optimize
                outputs = None
                batch_size = len(idx)
                opt_weights = [zz.clone().reshape(1, *zz.shape).expand(batch_size, *zz.shape) for zz in self.model.z0]
                if 'obstacles_map' in self.prob_features:
                    X_cnn = np.zeros((len(idx), 3,self.problem.H,self.problem.W))
                    for idx_ii, idx_val in enumerate(idx):
                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                        X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                    cnn_inputs = Variable(torch.from_numpy(X_cnn)).float().to(device=self.device)

                    outputs = model(cnn_inputs, ff_inputs, opt_weights)
                else:
                    outputs = model(ff_inputs, opt_weights)

                loss = training_loss(outputs, labels).float().to(device=self.device)
                class_guesses = torch.argmax(outputs,1)
                accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    rand_idx = list(np.arange(0,X.shape[0]-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    ff_inputs = Variable(torch.from_numpy(X[test_inds,:])).float().to(device=self.device)
                    labels = Variable(torch.from_numpy(Y[test_inds])).long().to(device=self.device)

                    # forward + backward + optimize
                    opt_weights = [zz.clone().reshape(1, *zz.shape).expand(TEST_BATCH_SIZE, *zz.shape) for zz in self.model.z0]
                    if type(model) is CNNet:
                        X_cnn = np.zeros((len(test_inds), 3,self.problem.H,self.problem.W))
                        for idx_ii, idx_val in enumerate(test_inds):
                            prob_params = {}
                            for k in params:
                                prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                            X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                        cnn_inputs = Variable(torch.from_numpy(X_cnn)).float().to(device=self.device)
                        outputs = model(cnn_inputs, ff_inputs, opt_weights)
                    else:
                        outputs = model(ff_inputs, opt_weights)

                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float())

                    writer.add_scalar('Loss/train', running_loss / float(self.training_params['CHECKPOINT_AFTER']) / float(BATCH_SIZE), itr)
                    writer.add_scalar('Loss/test', loss / float(TEST_BATCH_SIZE), itr)
                    writer.add_scalar('Loss/accuracy', accuracy.item(), itr)
                    running_loss = 0.

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.z0, self.model_fn)
                    verbose and print('Saved model at {}'.format(self.model_fn))

                itr += 1
            verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        torch.save(model.z0, self.model_fn)
        print('Saved model at {}'.format(self.model_fn))

        print('Done training')

    def forward(self, prob_params, solver=cp.MOSEK, max_evals=16):
        y_guesses = np.zeros((self.n_evals**self.problem.n_obs, self.n_y), dtype=int)

        # Compute forward pass for each obstacle and save the top
        # n_eval's scoring strategies in ind_max
        total_time = 0.   # start timing forward passes of network

        if 'obstacles_map' in self.prob_features:
            features = self.problem.construct_features(prob_params, self.prob_features, ii_obs=None)
            inpt = torch.from_numpy(features).repeat(self.problem.n_obs,1).float()

            cnn_features = np.zeros((self.problem.n_obs, 3,self.problem.H,self.problem.W))
            for ii_obs in range(self.problem.n_obs):
              cnn_features[ii_obs] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=ii_obs)
            cnn_inpt = torch.from_numpy(cnn_features).float()

            t0 = time.time()

            opt_weights = [zz.clone().reshape(1, *zz.shape).expand(self.problem.n_obs, *zz.shape) for zz in self.model.z0]
            with torch.no_grad():
                scores = self.model(cnn_inpt, inpt, opt_weights).cpu().detach().numpy()[:]
        else:
            features = self.problem.construct_features(prob_params, self.prob_features, ii_obs=ii_obs)
            inpt = Variable(torch.from_numpy(features)).unsqueeze(0).float()
            t0 = time.time()
            with torch.no_grad():
                scores = self.model(inpt, opt_weights).cpu().detach().numpy()[:].squeeze(0)
        torch.cuda.synchronize()
        total_time += (time.time()-t0)

        ind_max = np.argsort(scores, axis=1)[:,-self.n_evals:][:,::-1]

        # Loop through strategy dictionary once
        # Save ii'th stratey in obs_strats dictionary
        obs_strats = {}
        uniq_idxs = np.unique(ind_max)

        for ii, idx in enumerate(uniq_idxs):
            obs_strats[idx] = self.binary_dict[idx]

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

        prob_success, cost, n_evals, optvals = False, np.Inf, max_evals, None
        for ii, str_tuple in enumerate(strategy_tuples):
            y_guess = -np.ones((4*self.problem.n_obs, self.problem.N-1))
            for ii_obs in range(self.problem.n_obs):
                # rows of ind_max correspond to ii_obs, column to desired strategy
                y_obs = obs_strats[ind_max[ii_obs, str_tuple[ii_obs]]]
                y_guess[4*ii_obs:4*(ii_obs+1)] = np.reshape(y_obs, (4,self.problem.N-1))
            if (y_guess < 0).any():
                print("Strategy was not correctly found!")
                return False, np.Inf, total_time, n_evals, optvals

            prob_success, cost, solve_time, optvals = self.problem.solve_pinned(prob_params, y_guess, solver=solver)

            total_time += solve_time
            n_evals = ii+1
            if prob_success:
                break

        return prob_success, cost, total_time, n_evals, optvals
