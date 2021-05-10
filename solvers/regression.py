import os
import cvxpy as cp
import pickle
import numpy as np
import pdb
import time
import random
import sys
import torch

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sigmoid
from datetime import datetime

sys.path.insert(1, os.environ['CoCo'])
sys.path.insert(1, os.path.join(os.environ['CoCo'], 'pytorch'))

from core import Problem, Solver 
from pytorch.models import FFNet

class Regression(Solver):
    def __init__(self, system, problem, prob_features):
        """Constructor for Regression class.

        Args:
            system: string for system name e.g. 'cartpole'
            problem: Problem object for system
            prob_features: list of features to be used
        """
        super().__init__()
        self.system = system
        self.problem = problem
        self.prob_features = prob_features

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None

        # training parameters
        self.training_params = {}
        self.training_params['TRAINING_ITERATIONS'] = int(1500)
        self.training_params['BATCH_SIZE'] = 128
        self.training_params['CHECKPOINT_AFTER'] = int(20000)
        self.training_params['SAVEPOINT_AFTER'] = int(30000)
        self.training_params['TEST_BATCH_SIZE'] = 320

    def construct_strategies(self, n_features, train_data, test_data=None):
        """ Reads in training data and constructs strategy dictonary
            TODO(acauligi): be able to compute strategies for test-set too
        """
        self.n_features = n_features

        p_train = train_data[0]
        y_train = train_data[-3]
        for k in p_train.keys():
            self.num_train = len(p_train[k])

        ## TODO(acauligi): add support for combining p_train & p_test correctly
        ## to be able to generate strategies for train and test params here
        # p_test = None
        # y_test = np.empty((*y_train.shape[:-1], 0))   # Assumes y_train is 3D tensor
        # if test_data:
        #   p_test, y_test = test_data[:2]
        #   for k in p_test.keys():
        #     self.num_test = len(p_test[k])
        # num_probs = self.num_train + self.num_test
        # self.Y = np.dstack((Y_train, Y_test))         # Stack Y_train and Y_test along dim=2
        num_probs = self.num_train
        params = p_train
        self.Y = y_train

        self.n_y = self.Y[0].size
        self.y_shape = self.Y[0].shape
        self.features = np.zeros((num_probs, self.n_features))
        self.labels = np.zeros((num_probs, self.n_y))

        for ii in range(num_probs):
            # TODO(acauligi): check if transpose necessary with new pickle save format for Y
            self.labels[ii] = np.reshape(self.Y[ii,:,:].T, (self.n_y))

            prob_params = {}
            for k in params:
                prob_params[k] = params[k][ii]
            self.features[ii] = self.problem.construct_features(prob_params, self.prob_features)

    def setup_network(self, depth=3, neurons=32, device_id=0):
        self.device = torch.device('cuda:{}'.format(device_id))

        ff_shape = [self.n_features]
        for ii in range(depth):
            ff_shape.append(neurons)
        ff_shape.append(self.n_y)

        self.model = FFNet(ff_shape, activation='relu', cond_type='all_weights').to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'regression_{}_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(self.system, now)

    def load_network(self, fn_regressor_model):
        if os.path.exists(fn_regressor_model):
            print('Loading presaved regression model from {}'.format(fn_regressor_model))
            saved_params = list(torch.load(fn_regressor_model).values())
            for ii in range(len(saved_params)):
                self.model.vars[ii].data.copy_(saved_params[ii])
            self.model_fn = fn_regressor_model

    def train(self, verbose=True):
        # grab training params
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        model = self.model

        X = self.features[:self.num_train]
        Y = self.labels[:self.num_train]

        # See: https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/45
        training_loss = torch.nn.BCEWithLogitsLoss()
        opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)

        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(0,X.shape[0]-1))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii,idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                inputs = Variable(torch.from_numpy(X[idx,:])).float().to(device=self.device)
                y_true = Variable(torch.from_numpy(Y[idx,:])).float().to(device=self.device)

                # forward + backward + optimize
                outputs = model(inputs, model.z0)
                loss = training_loss(outputs, y_true).float().to(device=self.device)
                loss.backward()
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    rand_idx = list(np.arange(0,X.shape[0]-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    inputs = Variable(torch.from_numpy(X[test_inds,:])).float().to(device=self.device)
                    y_out = Variable(torch.from_numpy(Y[test_inds])).float().to(device=self.device)

                    # forward + backward + optimize
                    outputs = model(inputs, model.z0)
                    loss = training_loss(outputs, y_out).float().to(device=self.device)
                    outputs = Sigmoid()(outputs).round()
                    accuracy = [float(all(torch.eq(outputs[ii],y_out[ii]))) for ii in range(TEST_BATCH_SIZE)]
                    accuracy = np.mean(accuracy)
                    verbose and print("loss:   "+str(loss.item()) + " , acc: " + str(accuracy))

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    verbose and print('Saved model at {}'.format(self.model_fn))
                    # writer.add_scalar('Loss/train', running_loss, epoch)

                itr += 1
            verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        torch.save(model.state_dict(), self.model_fn)
        print('Saved model at {}'.format(self.model_fn))

        print('Done training')

    def forward(self, prob_params, solver=cp.MOSEK):
        features = self.problem.construct_features(prob_params, self.prob_features)
        inpt = Variable(torch.from_numpy(features)).float().unsqueeze(0).to(device=self.device)
        t0 = time.time()
        out = self.model(inpt, self.model.z0).cpu().detach()
        torch.cuda.synchronize()
        total_time = time.time()-t0
        y_guess = Sigmoid()(out).round().numpy()[0,:]

        # weirdly need to reshape in reverse order of cvxpy variable shape
        y_guess = np.reshape(y_guess, self.y_shape[::-1]).T

        prob_success, cost, optvals = False, np.Inf, None
        prob_success, cost, solve_time, optvals = self.problem.solve_pinned(prob_params, y_guess, solver)
        total_time += solve_time
        return prob_success, cost, total_time, optvals
