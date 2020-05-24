import os
import sys
sys.path.insert(1, os.path.join(os.environ['MLOPT'], 'pytorch'))

import pdb
import mosek
import cvxpy as cp
import numpy as np

import pdb
import h5py

from models import FFNet, BnBCNN
import time
import random
import string
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sigmoid
from datetime import datetime

class Optimizer():
  def __init__(self, n_files=20):
    pass

  def construct_bin_prob(self):
    pass
  
  def construct_mlopt_prob(self):
    pass

  def solve_bin_prob_with_params(self, params, solver=cp.MOSEK):
    # Takes problem params and solves binary problem 
    for k,v in self.bin_prob_parameters.items():
      v.value = params[k]

    prob_success, cost = False, np.Inf
    if solver == cp.MOSEK:
      msk_param_dict = {}
      msk_param_dict['MSK_IPAR_PRESOLVE_USE'] = 0
      msk_param_dict['MSK_IPAR_NUM_THREADS'] = 1

      self.bin_prob.solve(solver=solver, mosek_params=msk_param_dict)

    prob_success, cost, solve_time = False, np.Inf, np.Inf
    solve_time = self.bin_prob.solver_stats.solve_time
    if self.bin_prob.status == 'optimal':
      prob_success = True
      cost = self.bin_prob.value

    # Clear parameter values after solving
    for k,v in self.bin_prob_parameters.items():
      v.value = None 
    return prob_success, cost, solve_time

  def solve_mlopt_prob_with_params(self, params, y_guess, solver=cp.MOSEK):
    # Takes problem params and solves convex-ified problem 
    for k,v in self.mlopt_prob_parameters.items():
      v.value = params[k]

    prob_success, cost, solve_time = False, np.Inf, np.Inf
    if solver == cp.MOSEK:
      msk_param_dict = {}
      self.mlopt_prob.solve(solver=solver, mosek_params=msk_param_dict)
    elif solver == cp.OSQP:
      self.mlopt_prob.solve(solver=solver)

    solve_time = self.mlopt_prob.solver_stats.solve_time
    if self.mlopt_prob.status == 'optimal':
      prob_success = True
      cost = self.mlopt_prob.value

    # Clear parameter values after solving
    for k,v in self.mlopt_prob_parameters.items():
      v.value = None

    return prob_success, cost, solve_time

  def which_M(self, prob_idx):
    pass

  def construct_features(self, prob_idx):
    pass

  def construct_strategies(self):
    self.strategy_dict = {}
    self.training_labels = {}

    n_training_probs = int(round(self.training_batch_percentage * self.n_probs))
    self.features = np.zeros((n_training_probs, self.n_features))
    self.labels = np.zeros((n_training_probs, 1+self.n_y))
    self.n_strategies = 0

    for ii in range(n_training_probs):
      y_true = np.reshape(self.Y[:,:,ii].T, (self.n_y))
      if tuple(y_true) not in self.strategy_dict.keys():
        label = np.hstack((self.n_strategies,np.copy(y_true)))
        self.strategy_dict[tuple(y_true)] = label
        self.n_strategies += 1
      else:
        label = np.hstack((self.strategy_dict[tuple(y_true)][0], y_true))

      features = self.construct_features(ii) 
      self.training_labels[tuple(features)] = self.strategy_dict[tuple(y_true)]

      self.features[ii] = features
      self.labels[ii] =  label

  def setup_network(self, depth=3, neurons=32): 
    ff_shape = [self.n_features]
    for ii in range(depth):
      ff_shape.append(neurons)

    ff_shape.append(self.n_strategies)
    self.model_classifier = FFNet(ff_shape, activation=torch.nn.ReLU()).cuda()

    ff_shape.pop()
    ff_shape.append(int(self.labels.shape[1]-1))

    self.model_regressor = FFNet(ff_shape, activation=torch.nn.ReLU()).cuda()

    if os.path.exists(self.fn_classifier_model):
      print('Loading presaved classifier model from {}'.format(self.fn_classifier_model))
      self.model_classifier.load_state_dict(torch.load(self.fn_classifier_model))
    if os.path.exists(self.fn_regressor_model):
      print('Loading presaved regressor model from {}'.format(self.fn_regressor_model))
      self.model_regressor.load_state_dict(torch.load(self.fn_regressor_model))

  def train_regressor(self):
    # grab training params
    BATCH_SIZE = self.training_params['BATCH_SIZE']
    TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
    BATCH_SIZE = self.training_params['BATCH_SIZE']
    CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
    SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
    TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

    model = self.model_regressor

    n_training_probs = int(round(self.training_batch_percentage * self.n_probs))
    X = self.features
    Y = self.labels[:,1:]

    # See: https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/45
    training_loss = torch.nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    itr = 1
    for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
      t0 = time.time()
      running_loss = 0.0
      rand_idx = np.arange(0,X.shape[0]-1)
      random.shuffle(rand_idx)
      indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

      for ii,idx in enumerate(indices):
        # zero the parameter gradients
        opt.zero_grad()

        inputs = Variable(torch.from_numpy(X[idx,:])).float().cuda()
        y_true = Variable(torch.from_numpy(Y[idx,:])).float().cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = training_loss(outputs, y_true).float().cuda()
        loss.backward()
        opt.step()

        # print statistics\n",
        running_loss += loss.item()
        if itr % CHECKPOINT_AFTER == 0:
          rand_idx = list(np.arange(0,X.shape[0]-1))
          random.shuffle(rand_idx)
          test_inds = rand_idx[:TEST_BATCH_SIZE]
          inputs = Variable(torch.from_numpy(X[test_inds,:])).float().cuda()
          y_out = Variable(torch.from_numpy(Y[test_inds])).float().cuda()

          # forward + backward + optimize
          outputs = model(inputs)
          loss = training_loss(outputs, y_out).float().cuda()
          outputs = Sigmoid()(outputs).round()
          accuracy = [float(all(torch.eq(outputs[ii],y_out[ii]))) for ii in range(TEST_BATCH_SIZE)]
          accuracy = np.mean(accuracy)
          print("loss:   "+str(loss.item()) + " , acc: " + str(accuracy))

        if itr % SAVEPOINT_AFTER == 0:
          torch.save(model.state_dict(), self.fn_regressor_model)
          print('Saved model at {}'.format(self.fn_regressor_model))
          # writer.add_scalar('Loss/train', running_loss, epoch)

        itr += 1

      print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

    torch.save(model.state_dict(), self.fn_regressor_model)
    print('Saved model at {}'.format(self.fn_regressor_model))

  def train_classifier(self):
    # grab training params
    BATCH_SIZE = self.training_params['BATCH_SIZE']
    TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
    BATCH_SIZE = self.training_params['BATCH_SIZE']
    CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
    SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
    TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

    model = self.model_classifier
    
    n_training_probs = int(round(self.training_batch_percentage * self.n_probs))
    X = self.features
    Y = self.labels[:,0]

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

        inputs = Variable(torch.from_numpy(X[idx,:])).float().cuda()
        labels = Variable(torch.from_numpy(Y[idx])).long().cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = training_loss(outputs, labels).float().cuda()
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
          inputs = Variable(torch.from_numpy(X[test_inds,:])).float().cuda()
          labels = Variable(torch.from_numpy(Y[test_inds])).long().cuda()

          # forward + backward + optimize
          outputs = model(inputs)
          loss = training_loss(outputs, labels).float().cuda()
          class_guesses = torch.argmax(outputs,1)
          accuracy = torch.mean(torch.eq(class_guesses,labels).float())
          print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

        if itr % SAVEPOINT_AFTER == 0:
          torch.save(model.state_dict(), self.fn_classifier_model)
          print('Saved model at {}'.format(self.fn_classifier_model))
          # writer.add_scalar('Loss/train', running_loss, epoch)

        itr += 1

      print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

    torch.save(model.state_dict(), self.fn_classifier_model)
    print('Saved model at {}'.format(self.fn_classifier_model))

    print('Done training')

  def solve_with_classifier(self, prob_idx, solver=cp.MOSEK):
    features = self.construct_features(prob_idx)
    inpt = Variable(torch.from_numpy(features)).float().cuda()
    scores = self.model_classifier(inpt).cpu().detach().numpy()[:]
    ind_max = np.argsort(scores)[-self.n_evals:][::-1]

    y_guesses = np.zeros((self.n_evals, self.n_y), dtype=int)

    for k,v in self.training_labels.items():
      for ii,idx in enumerate(ind_max):
        if v[0] == idx:
          y_guesses[ii] = v[1:]

    prob_success, cost, total_time, n_evals = False, np.Inf, 0., len(y_guesses)
    for ii,idx in enumerate(ind_max):
      y_guess = y_guesses[ii]
      prob_success, cost, solve_time = self.solve_mlopt_prob_with_idx(prob_idx, y_guess, solver=solver)

      total_time += solve_time
      n_evals = ii+1
      if prob_success:
        prob_success = True
        break
    return prob_success, cost, total_time, n_evals

  def solve_with_regressor(self, prob_idx, solver=cp.MOSEK):
    features = self.construct_features(prob_idx)
    inpt = Variable(torch.from_numpy(features)).float().cuda()
    out = self.model_regressor(inpt).cpu().detach()
    y_guess = Sigmoid()(out).round().numpy()[:]

    prob_success, cost, solve_time = self.solve_mlopt_prob_with_idx(prob_idx, y_guess, solver=solver)
    return prob_success, cost, solve_time
