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

class CoCo(Solver):
    def __init__(self, system, problem, prob_features, n_evals=10):
        """Constructor for CoCo class.

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

        # training parameters
        self.training_params = {}
        self.training_params['TRAINING_ITERATIONS'] = int(1500)
        self.training_params['BATCH_SIZE'] = 64
        self.training_params['CHECKPOINT_AFTER'] = int(1000)
        self.training_params['SAVEPOINT_AFTER'] = int(30000)
        self.training_params['TEST_BATCH_SIZE'] = 32

    def construct_strategies(self, n_features, train_data, test_data=None):
        """ Reads in training data and constructs strategy dictonary
            TODO(acauligi): be able to compute strategies for test-set too
        """
        self.n_features = n_features
        self.strategy_dict = {}

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
        self.labels = np.zeros((num_probs, 1+self.n_y))
        self.n_strategies = 0

        for ii in range(num_probs):
            # TODO(acauligi): check if transpose necessary with new pickle save format for Y
            y_true = np.reshape(self.Y[ii,:,:], (self.n_y))

            if tuple(y_true) not in self.strategy_dict.keys():
                self.strategy_dict[tuple(y_true)] = np.hstack((self.n_strategies,np.copy(y_true)))
                self.n_strategies += 1
            self.labels[ii] = self.strategy_dict[tuple(y_true)]

            prob_params = {}
            for k in params:
                prob_params[k] = params[k][ii]

            self.features[ii] = self.problem.construct_features(prob_params, self.prob_features)

    def setup_network(self, depth=3, neurons=32, device_id=0):
        self.device = torch.device('cuda:{}'.format(device_id))

        ff_shape = [self.n_features]
        for ii in range(depth):
            ff_shape.append(neurons)

        ff_shape.append(self.n_strategies)
        self.model = FFNet(ff_shape, activation=torch.nn.ReLU()).to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'CoCo_{}_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(self.system, now)

    def load_network(self, fn_classifier_model):
        if os.path.exists(fn_classifier_model):
            print('Loading presaved classifier model from {}'.format(fn_classifier_model))
            self.model.load_state_dict(torch.load(fn_classifier_model))
            self.model_fn = fn_classifier_model

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
        Y = self.labels[:self.num_train,0]

        training_loss = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.00001)

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
                labels = Variable(torch.from_numpy(Y[idx])).long().to(device=self.device)

                # forward + backward + optimize
                outputs = model(inputs)
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
                    inputs = Variable(torch.from_numpy(X[test_inds,:])).float().to(device=self.device)
                    labels = Variable(torch.from_numpy(Y[test_inds])).long().to(device=self.device)

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                    verbose and print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

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
        inpt = Variable(torch.from_numpy(features)).float().to(device=self.device)
        t0 = time.time()
        scores = self.model(inpt).cpu().detach().numpy()[:]
        torch.cuda.synchronize()
        total_time = time.time()-t0
        ind_max = np.argsort(scores)[-self.n_evals:][::-1]

        y_guesses = np.zeros((self.n_evals, self.n_y), dtype=int)

        num_probs = self.num_train
        for ii,idx in enumerate(ind_max):
            for jj in range(num_probs):
                # first index of training label is that strategy's idx
                label = self.labels[jj]
                if label[0] == idx:
                    # remainder of training label is that strategy's binary pin
                    y_guesses[ii] = label[1:]
                    break

        prob_success, cost, n_evals, optvals = False, np.Inf, len(y_guesses), None
        for ii,idx in enumerate(ind_max):
            y_guess = y_guesses[ii]

            # weirdly need to reshape in reverse order of cvxpy variable shape
            y_guess = np.reshape(y_guess, self.y_shape)

            prob_success, cost, solve_time, optvals = self.problem.solve_pinned(prob_params, y_guess, solver)

            total_time += solve_time
            n_evals = ii+1
            if prob_success:
                break
        return prob_success, cost, total_time, n_evals, optvals
