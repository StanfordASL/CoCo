import os
import cvxpy as cp
import pickle
import numpy as np
import pdb
import time
import random
import sys
import torch

from datetime import datetime

sys.path.insert(1, os.environ['MLOPT'])

from core import Problem, Solver 

class KNN(Solver):
    def __init__(self, system, problem, prob_features, knn=5):
        """Constructor for KNN class.

        Args:
            system: string for system name e.g. 'cartpole'
            problem: Problem object for system
            prob_features: list of features to be used
        """
        super().__init__()
        self.system = system
        self.problem = problem
        self.prob_features = prob_features
        self.knn = knn

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None

    def train(self, n_features, train_data, test_data=None):
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
        num_probs = self.num_train
        params = p_train
        self.Y = y_train

        self.n_y = self.Y[0].size
        self.y_shape = self.Y[0].shape
        self.features = np.zeros((num_probs, self.n_features))
        self.labels = np.zeros((num_probs, 1+self.n_y))
        self.n_strategies = 0

        str_dict = {}
        for ii in range(num_probs):
            # TODO(acauligi): check if transpose necessary with new pickle save format for Y
            y_true = np.reshape(self.Y[ii,:,:], (self.n_y))

            if tuple(y_true) not in self.strategy_dict.keys():
                self.strategy_dict[tuple(y_true)] = np.hstack((self.n_strategies,np.copy(y_true)))
                self.n_strategies += 1
            self.labels[ii] = self.strategy_dict[tuple(y_true)]

            idx = int(self.labels[ii,0])
            if idx in str_dict.keys():
                str_dict[idx] += [ii]
            else:
                str_dict[idx] = [ii]

            prob_params = {}
            for k in params:
                prob_params[k] = params[k][ii]

            self.features[ii] = self.problem.construct_features(prob_params, self.prob_features)

        self.centroids = np.zeros((self.n_strategies, self.features.shape[1]))
        for ii in range(self.n_strategies):
            self.centroids[ii] = np.mean(self.features[str_dict[ii]], axis=0)

    def forward(self, prob_params, solver=cp.MOSEK):
        t0 = time.time()
        features = torch.from_numpy(self.problem.construct_features(prob_params, self.prob_features)).unsqueeze(0)
        ind_max = torch.argsort(torch.cdist(features, torch.from_numpy(self.centroids))).numpy()[0]
        ind_max = ind_max[:self.knn]
        total_time = time.time()-t0

        y_guesses = np.zeros((self.knn, self.n_y), dtype=int)
        for ii,idx in enumerate(ind_max):
            jj = np.where(self.labels[:,0] == idx)[0]
            y_guesses[ii] = self.labels[jj,1:]

        prob_success, cost, n_evals, optvals = False, np.Inf, len(y_guesses), None
        for ii,idx in enumerate(ind_max):
            y_guess = y_guesses[ii]

            # weirdly need to reshape in reverse order of cvxpy variable shape
            y_guess = np.reshape(y_guess, self.y_shape)

            prob_success, cost, solve_time, optvals = self.problem.solve_pinned(prob_params, y_guess, solver=solver)

            total_time += solve_time
            n_evals = ii+1
            if prob_success:
                break
        return prob_success, cost, total_time, n_evals, optvals
