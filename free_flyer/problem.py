import os
import sys
sys.path.insert(1, os.path.join(os.environ['MLOPT'], 'pytorch'))

import pdb
import mosek
import cvxpy as cp
import numpy as np

import pdb
import h5py
import itertools

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

system = "free_flyer"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Optimizer():
  def __init__(self, n_files=20):
    self.training_batch_percentage = 0.9

    self.solve_time = np.array([], dtype=float) 
    self.J = np.array([], dtype=float) 
    self.node_count = np.array([], dtype=int) 

    for n in range(n_files):
      fn = os.path.join(os.environ["MLOPT"], system, 'data/testdata{}.h5'.format(1))
      f = h5py.File(fn, 'r')
      if n == 0:
        self.N = f['N'][()]
        self.n = int(f['X'][()].T[:,:,0].shape[0]/2)
        self.m = int(f['U'][()].T[:,:,0].shape[0])
        self.n_obs = int(f['O'][()].T[:,:,0].shape[1]) 
        n_y = f['Y'][()].T[:,:,0].shape[0]

        self.X = np.empty((2*self.n,self.N,0), dtype=float)
        self.U = np.empty((self.m,self.N-1,0), dtype=float)
        self.Y = np.empty((n_y,self.N-1,0), dtype=int)
        self.O = np.empty((4,self.n_obs,0), dtype=float)
        self.X0 = np.empty((2*self.n,0), dtype=float)
        self.Xg = np.empty((2*self.n,0), dtype=float)

        self.Ak = f['Ak'][()].T
        self.Bk = f['Bk'][()].T
        self.Q = f['Q'][()]
        self.R = f['R'][()]
        self.posmin = f['posmin'][()]
        self.posmax = f['posmax'][()]
        self.velmin = f['velmin'][()]
        self.velmax = f['velmax'][()]
        self.umin = f['umin'][()]
        self.umax = f['umax'][()]

      self.X = np.dstack((self.X, f['X'][()].T))
      self.U = np.dstack((self.U, f['U'][()].T))
      self.Y = np.dstack((self.Y, f['Y'][()].T))
      self.O = np.dstack((self.O, f["O"][()].T))
      self.X0 = np.hstack((self.X0, f['X0'][()].T))
      self.Xg = np.hstack((self.Xg, f['Xg'][()].T))

      self.solve_time = np.append(self.solve_time, f['solve_time'][()])
      self.J = np.append(self.J, f['J'][()])
      self.node_count = np.append(self.node_count, f['node_count'][()])

      f.close()

    self.n_probs = self.X.shape[-1]

    self.prob_features = ["X0", "obstacles"]
    self.n_features = int(2*self.n + 4*self.n_obs + self.n_obs) 
    self.strategy_dict = {}
    self.training_labels = np.zeros((self.n_features, self.n_probs))
    self.n_strategies = 0
    self.n_evals = 2 

    # training parameters
    self.training_params = {}
    self.training_params['TRAINING_ITERATIONS'] = int(1500)
    self.training_params['BATCH_SIZE'] = 128
    self.training_params['CHECKPOINT_AFTER'] = int(1000)
    self.training_params['SAVEPOINT_AFTER'] = int(30000)
    self.training_params['TEST_BATCH_SIZE'] = 320

    # file names for PyTorch models
    now = datetime.now().strftime('%Y%m%d_%H%M')
    fn_model = 'mlopt_model_{}_{}_{}.pt'
    fn_model = os.path.join(os.getcwd(), fn_model)
    self.fn_regressor_model = fn_model.format(system, now, 'regressor')
    self.fn_classifier_model = fn_model.format(system, now, 'classifier')

    self.construct_bin_prob()
    self.construct_mlopt_prob()

    # self.construct_strategies()
    # self.setup_network()

  def construct_bin_prob(self):
    cons = []

    # Variables
    x = cp.Variable((2*self.n,self.N)) # state
    u = cp.Variable((self.m,self.N-1))  # control
    y = cp.Variable((4*self.n_obs,self.N-1), boolean=True)

    # Parameters
    x0 = cp.Parameter(2*self.n)
    xg = cp.Parameter(2*self.n)
    obstacles = cp.Parameter((4, self.n_obs))
    self.bin_prob_parameters = {'x0': x0, 'xg': xg, 'obstacles': obstacles} 

    cons += [x[:,0] == x0]

    # Dynamics constraints
    for ii in range(self.N-1):
      cons += [x[:,ii+1] - (self.Ak @ x[:,ii] + self.Bk @ u[:,ii]) == np.zeros(2*self.n)]

    M = 100. # big M value
    for i_obs in range(self.n_obs):
      for i_dim in range(self.n):
        o_min = obstacles[self.n*i_dim,i_obs]
        o_max = obstacles[self.n*i_dim+1,i_obs]

        for i_t in range(self.N-1):
          yvar_min = 4*i_obs + self.n*i_dim
          yvar_max = 4*i_obs + self.n*i_dim + 1

          cons += [x[i_dim,i_t+1] <= o_min + M*y[yvar_min,i_t]]
          cons += [-x[i_dim,i_t+1] <= -o_max + M*y[yvar_max,i_t]]

      for i_t in range(self.N-1):
        yvar_min, yvar_max = 4*i_obs, 4*(i_obs+1)
        cons += [sum([y[ii,i_t] for ii in range(yvar_min,yvar_max)]) <= 3]

    # Region bounds
    for kk in range(self.N):
      for jj in range(self.n):
        cons += [self.posmin[jj] - x[jj,kk] <= 0]
        cons += [x[jj,kk] - self.posmax[jj] <= 0]
    
    # Velocity constraints
    for kk in range(self.N):
      for jj in range(self.n):
        cons += [self.velmin - x[self.n+jj,kk] <= 0]
        cons += [x[self.n+jj,kk] - self.velmax <= 0]
        
    # Control constraints
    for kk in range(self.N-1):
      cons += [cp.norm(u[:,kk]) <= self.umax]
    
    lqr_cost = 0.
    # l2-norm of lqr_cost
    for kk in range(self.N):
      lqr_cost += cp.quad_form(x[:,kk]-xg, self.Q)
    
    for kk in range(self.N-1):
      lqr_cost += cp.quad_form(u[:,kk], self.R)

    self.bin_prob = cp.Problem(cp.Minimize(lqr_cost), cons)
    return x, u, y

  def construct_mlopt_prob(self):
    cons = []

    # Variables
    x = cp.Variable((2*self.n,self.N)) # state
    u = cp.Variable((self.m,self.N-1))  # control

    # Parameters
    x0 = cp.Parameter(2*self.n)
    xg = cp.Parameter(2*self.n)
    obstacles = cp.Parameter((4, self.n_obs))
    y = cp.Parameter((4*self.n_obs,self.N-1)) 
    self.mlopt_prob_parameters = {'x0': x0, 'xg': xg,
      'obstacles': obstacles, 'y':y}

    cons += [x[:,0] == x0]

    # Dynamics constraints
    for ii in range(self.N-1):
      cons += [x[:,ii+1] - (self.Ak @ x[:,ii] + self.Bk @ u[:,ii]) == np.zeros(2*self.n)]

    M = 100. # big M value
    for i_obs in range(self.n_obs):
      for i_dim in range(self.n):
        o_min = obstacles[self.n*i_dim,i_obs]
        o_max = obstacles[self.n*i_dim+1,i_obs]

        for i_t in range(self.N-1):
          yvar_min = 4*i_obs + self.n*i_dim
          yvar_max = 4*i_obs + self.n*i_dim + 1

          cons += [x[i_dim,i_t+1] <= o_min + M*y[yvar_min,i_t]]
          cons += [-x[i_dim,i_t+1] <= -o_max + M*y[yvar_max,i_t]]

      for i_t in range(self.N-1):
        yvar_min, yvar_max = 4*i_obs, 4*(i_obs+1)
        cons += [sum([y[ii,i_t] for ii in range(yvar_min,yvar_max)]) <= 3]

    # Region bounds
    for kk in range(self.N):
      for jj in range(self.n):
        cons += [self.posmin[jj] - x[jj,kk] <= 0]
        cons += [x[jj,kk] - self.posmax[jj] <= 0]
    
    # Velocity constraints
    for kk in range(self.N):
      for jj in range(self.n):
        cons += [self.velmin - x[self.n+jj,kk] <= 0]
        cons += [x[self.n+jj,kk] - self.velmax <= 0]
        
    # Control constraints
    for kk in range(self.N-1):
      cons += [cp.norm(u[:,kk]) <= self.umax]
    
    lqr_cost = 0.
    # l2-norm of lqr_cost
    for kk in range(self.N):
      lqr_cost += cp.quad_form(x[:,kk]-xg, self.Q)
    
    for kk in range(self.N-1):
      lqr_cost += cp.quad_form(u[:,kk], self.R)

    self.mlopt_prob = cp.Problem(cp.Minimize(lqr_cost), cons)
    return x, u, y

  def solve_bin_prob(self, solver=cp.MOSEK):
    params = {'x0': self.X0[:,prob_idx],
                'xg': self.Xg[:,prob_idx],
                'obstacles': self.O[:,:,prob_idx]} 
    for k,v in self.bin_prob_parameters.items():
      v.value = params[k]

    prob_success, cost = False, np.Inf
    if solver == cp.MOSEK:
      msk_param_dict = {}
      self.bin_prob.solve(solver=solver, mosek_params=msk_param_dict)
      if self.bin_prob.status == 'Optimal':
        prob_success = True
        cost = self.bin_prob.value

    # Clear parameter values after solving
    for k,v in self.bin_prob_parameters.items():
      v.value = None 
    return prob_success, cost

  def solve_mlopt_prob(self, prob_idx, y_guess, solver=cp.MOSEK):
    params = {'x0': self.X0[:,prob_idx], 'xg': self.Xg[:,prob_idx],
      'obstacles': self.O[:,:,prob_idx], 'y': np.reshape(y_guess, self.Y[:,:,0].shape)} 

    for k,v in self.mlopt_prob_parameters.items():
      v.value = params[k]

    prob_success, cost = False, np.Inf
    if solver == cp.MOSEK:
      msk_param_dict = {}
      self.mlopt_prob.solve(solver=solver, mosek_params=msk_param_dict)
      if self.mlopt_prob.status == 'optimal':
        prob_success = True
        cost = self.mlopt_prob.value

    # Clear parameter values after solving
    for k,v in self.mlopt_prob_parameters.items():
      v.value = None
    return prob_success, cost 

  def which_M(self, prob_idx, eq_tol=1e-5, ineq_tol=1e-5):
    x = self.X[:,:,prob_idx]
    obstacles = self.O[:,:,prob_idx]
    violations = [] # list of obstacle big-M violations

    for i_obs in range(self.n_obs):
      curr_violations = [] # violations for current obstacle
      for i_t in range(self.N-1):
        for i_dim in range(self.n):
          o_min = obstacles[self.n*i_dim,i_obs]
          o_max = obstacles[self.n*i_dim+1,i_obs]

          for i_t in range(self.N-1):
            yvar_min = 4*i_obs + self.n*i_dim
            yvar_max = 4*i_obs + self.n*i_dim + 1

            if (x[i_dim,i_t+1] - o_min > ineq_tol):
              curr_violations.append(4*self.n_obs*i_t + yvar_min)
            if (-x[i_dim,i_t+1]  + o_max > ineq_tol): 
              curr_violations.append(4*self.n_obs*i_t + yvar_max)
      violations.append(curr_violations)
    return violations
  
  def construct_features(self, prob_idx):
    feature_vec = np.array([])
    x0, xg = self.X0[:,prob_idx], self.Xg[:,prob_idx]
    obstacles = self.O[:,:,prob_idx]
    for feature in self.prob_features:
      if feature == "X0":
        feature_vec = np.hstack((feature_vec, x0))
      elif feature == "obstacles":
        feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4*self.n_obs))))
      elif feature == "obstacles_map":
        print("obstacles_map feature not implemented yet!")
      else:
        print('Feature {} is unknown'.format(feature))
    return feature_vec

  def construct_strategies(self):
    n_y = 4*(self.N-1)
    self.strategy_dict = {}
    self.training_labels = {}

    n_training_probs = int(round(self.training_batch_percentage * self.n_probs))
    self.features = np.zeros((self.n_obs*n_training_probs, self.n_features))
    self.labels = np.zeros((self.n_obs*n_training_probs, 1+n_y))
    self.n_strategies = 0

    for ii in range(n_training_probs):
      violations = self.which_M(ii)
      Y = self.Y[:,:,ii]
      for ii_obs, obs_strat in enumerate(violations):
        if tuple(obs_strat) not in self.strategy_dict.keys():
          y_obs = np.reshape(Y[4*ii_obs:4*(ii_obs+1),:], (n_y))
          label = np.hstack((self.n_strategies, y_obs))
          self.strategy_dict[tuple(obs_strat)] = label
          self.n_strategies += 1
        else:
          label = self.strategy_dict[tuple(obs_strat)] 

        feature = np.zeros(self.n_obs)
        feature[ii_obs] = 1
        feature = np.hstack((feature, self.construct_features(ii)))
        self.training_labels[tuple(feature)] = self.strategy_dict[tuple(obs_strat)]
        self.features[self.n_obs*ii+ii_obs] = feature 
        self.labels[self.n_obs*ii+ii_obs] = label

  def setup_network(self, depth=3, neurons=32): 
    ff_shape = [self.n_features]
    for ii in range(depth):
      ff_shape.append(neurons)

    ff_shape.append(self.n_strategies)
    self.model_classifier = FFNet(ff_shape, activation=torch.nn.ReLU()).cuda()

    ff_shape.pop()
    n_y = 4*(self.N-1)
    ff_shape.append(n_y)

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

  def solve_with_classifier(self, prob_idx, max_evals=16):
    n_y = self.Y[:,:,0].size
    y_guesses = np.zeros((self.n_evals**self.n_obs, n_y), dtype=int)

    # Compute forward pass for each obstacle and save the top
    # n_eval's scoring strategies in ind_max
    prob_features = self.construct_features(prob_idx)
    ind_max = np.zeros((self.n_obs, self.n_evals), dtype=int)
    for ii_obs in range(self.n_obs):
      features= np.zeros(self.n_obs)
      features[ii_obs] = 1
      features = np.hstack((features, prob_features))

      inpt = Variable(torch.from_numpy(features)).float().cuda()
      scores = self.model_classifier(inpt).cpu().detach().numpy()[:]
      ind_max[ii_obs] = np.argsort(scores)[-self.n_evals:][::-1]

    # Loop through strategy dictionary once
    # Save i'th stratey in obs_strats dictionary
    obs_strats = {}
    uniq_idxs = np.unique(ind_max)
    for k,v in self.training_labels.items():
      for ii,idx in enumerate(uniq_idxs):
        if v[0] == idx:
          obs_strats[idx] = v[1:]
      if len(obs_strats) == uniq_idxs.size:
        # All strategies found 
        break

    # Generate Cartesian product of strategy combinations
    vv = [np.arange(0,self.n_evals) for _ in range(self.n_obs)]
    strategy_tuples = itertools.product(*vv)

    found_soln = False
    for ii, str_tuple in enumerate(strategy_tuples):
      if ii >= max_evals:
        break
      y_guess = -np.ones((4*self.n_obs, self.N-1))
      for ii_obs in range(self.n_obs):
        y_obs = obs_strats[ind_max[ii_obs, str_tuple[ii_obs]]]
        y_guess[4*ii_obs:4*(ii_obs+1)] = np.reshape(y_obs, (4,self.N-1))

      if (y_guess < 0).any():
        print("Strategy was not correctly found!")
        return

      y_guess = np.reshape(y_guess, (y_guess.size))

      prob_success, cost = self.solve_mlopt_prob(prob_idx, y_guess, solver=cp.MOSEK)
      if prob_success:
        print("Succeeded!")
        found_soln = True
        break

    if not found_soln:
      print("Failed!")