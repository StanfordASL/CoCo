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

system = "cart-pole"
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
        self.n = int(f['X'][()].T[:,:,0].shape[0])
        self.m = int(f['U'][()].T[:,:,0].shape[0])
        n_y = f['Y'][()].T[:,:,0].shape[0]

        self.X = np.empty((self.n,self.N,0), dtype=float)
        self.U = np.empty((self.m,self.N-1,0), dtype=float)
        self.Y = np.empty((n_y,self.N-1,0), dtype=int)
        self.X0 = np.empty((self.n,0), dtype=float)
        self.Xg = np.empty((self.n,0), dtype=float)

        self.Ak = f['Ak'][()].T
        self.Bk = f['Bk'][()].T
        self.Q = f['Q'][()]
        self.R = f['R'][()]
        self.x_min = f['x_min'][()]
        self.x_max = f['x_max'][()]
        self.uc_min = f['uc_min'][()]
        self.uc_max = f['uc_max'][()]
        self.sc_min = f['sc_min'][()]
        self.sc_max = f['sc_max'][()]
        self.delta_min = f['delta_min'][()]
        self.delta_max = f['delta_max'][()]
        self.ddelta_min = f['ddelta_min'][()]
        self.ddelta_max = f['ddelta_max'][()]
        self.dh = f['dh'][()]
        self.g = f['g'][()]
        self.l = f['l'][()]
        self.mc = f['mc'][()]
        self.mp = f['mp'][()]
        self.kappa = f['kappa'][()]
        self.nu = f['nu'][()]
        self.dist = f['dist'][()]

      self.X = np.dstack((self.X, f['X'][()].T))
      self.U = np.dstack((self.U, f['U'][()].T))
      self.Y = np.dstack((self.Y, f['Y'][()].T))
      self.X0 = np.hstack((self.X0, f['X0'][()].T))
      self.Xg = np.hstack((self.Xg, f['Xg'][()].T))

      self.solve_time = np.append(self.solve_time, f['solve_time'][()])
      self.J = np.append(self.J, f['J'][()])
      self.node_count = np.append(self.node_count, f['node_count'][()])
      f.close()
    self.n_probs = self.X.shape[-1]
    
    # Strategy dictionary variables
    self.prob_features = ['X0', 'Xg', 'dist_to_goal', 'delta2_0', 'delta3_0', 'delta2_g', 'delta3_g']
    self.n_features = 13
    self.strategy_dict = {}
    self.training_labels = np.zeros((self.n_features, self.n_probs))
    self.n_strategies = 0
    self.n_evals = 10
    
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

    self.construct_strategies()
    self.setup_network()

  def construct_bin_prob(self):
    cons = []

    x = cp.Variable((self.n,self.N))
    u = cp.Variable((self.m, self.N-1))
    sc = u[1:,:]
    y = cp.Variable((4, self.N-1), boolean=True)

    x0 = cp.Parameter(self.n)
    xg = cp.Parameter(self.n)
    self.bin_prob_parameters = {'x0': x0, 'xg': xg}

    # Initial condition
    cons += [x[:,0] == x0] 

    # Dynamics constraints
    for kk in range(self.N-1):
      cons += [x[:,kk+1] - (self.Ak @ x[:,kk] + self.Bk @ u[:,kk]) == np.zeros(self.n)]

    # State and control constraints
    for kk in range(self.N):
      cons += [self.x_min - x[:,kk] <= np.zeros(self.n)] 
      cons += [x[:,kk] - self.x_max <= np.zeros(self.n)]

    for kk in range(self.N-1):
      cons += [self.uc_min - u[0,kk] <= 0.]
      cons += [u[0,kk] - self.uc_max <= 0]

    # Binary variable constraints
    for kk in range(self.N-1):
      for jj in range(2):
        if jj == 0:
          d_k    = -x[0,kk] + self.l*x[1,kk] - self.dist
          dd_k   = -x[2,kk] + self.l*x[3,kk]
        else:
          d_k    =  x[0,kk] - self.l*x[1,kk] - self.dist
          dd_k   =  x[2,kk] - self.l*x[3,kk]

        y_l, y_r = y[2*jj:2*jj+2,kk]
        d_min, d_max = self.delta_min[jj], self.delta_max[jj]
        dd_min, dd_max = self.ddelta_min[jj], self.ddelta_max[jj]
        f_min, f_max = self.sc_min[jj], self.sc_max[jj]

        # Eq. (26a)
        cons += [d_min*(1-y_l) <= d_k]
        cons += [d_k <= d_max*y_l]

        # Eq. (26b)
        cons += [f_min*(1-y_r) <= self.kappa*d_k + self.nu*dd_k]
        cons += [self.kappa*d_k + self.nu*dd_k <= f_max*y_r]

        # Eq. (27)
        cons += [self.nu*dd_max*(y_l-1) <= sc[jj,kk] - self.kappa*d_k - self.nu*dd_k]
        cons += [sc[jj,kk] - self.kappa*d_k - self.nu*dd_k <= f_min*(y_r-1)]

        cons += [-sc[jj,kk] <= 0]
        cons += [sc[jj,kk] <= f_max*y_l]
        cons += [sc[jj,kk] <= f_max*y_r]

    # LQR cost
    lqr_cost = 0.
    for kk in range(self.N):
      lqr_cost += cp.quad_form(x[:,kk]-xg, self.Q)
    for kk in range(self.N-1):
      lqr_cost += cp.quad_form(u[:,kk],self.R)

    self.bin_prob = cp.Problem(cp.Minimize(lqr_cost), cons)
    return x, u, y

  def construct_mlopt_prob(self):
    cons = []

    x = cp.Variable((self.n,self.N))
    u = cp.Variable((self.m, self.N-1))
    sc = u[1:,:]

    x0 = cp.Parameter(self.n)
    xg = cp.Parameter(self.n)
    y = cp.Parameter((4, self.N-1)) 
    self.mlopt_prob_parameters = {'x0': x0, 'xg': xg, 'y': y}

    # Initial condition
    cons += [x[:,0] == x0] 

    # Dynamics constraints
    for kk in range(self.N-1):
      cons += [x[:,kk+1] - (self.Ak @ x[:,kk] + self.Bk @ u[:,kk]) == np.zeros(self.n)]

    # State and control constraints
    for kk in range(self.N):
      cons += [self.x_min - x[:,kk] <= np.zeros(self.n)] 
      cons += [x[:,kk] - self.x_max <= np.zeros(self.n)]

    for kk in range(self.N-1):
      cons += [self.uc_min - u[0,kk] <= 0.]
      cons += [u[0,kk] - self.uc_max <= 0]

    # Binary variable constraints
    for kk in range(self.N-1):
      for jj in range(2):
        if jj == 0:
          d_k    = -x[0,kk] + self.l*x[1,kk] - self.dist
          dd_k   = -x[2,kk] + self.l*x[3,kk]
        else:
          d_k    =  x[0,kk] - self.l*x[1,kk] - self.dist
          dd_k   =  x[2,kk] - self.l*x[3,kk]

        y_l, y_r = y[2*jj:2*jj+2,kk]
        d_min, d_max = self.delta_min[jj], self.delta_max[jj]
        dd_min, dd_max = self.ddelta_min[jj], self.ddelta_max[jj]
        f_min, f_max = self.sc_min[jj], self.sc_max[jj]

        # Eq. (26a)
        cons += [d_min*(1-y_l) <= d_k]
        cons += [d_k <= d_max*y_l]

        # Eq. (26b)
        cons += [f_min*(1-y_r) <= self.kappa*d_k + self.nu*dd_k]
        cons += [self.kappa*d_k + self.nu*dd_k <= f_max*y_r]

        # Eq. (27)
        cons += [self.nu*dd_max*(y_l-1) <= sc[jj,kk] - self.kappa*d_k - self.nu*dd_k]
        cons += [sc[jj,kk] - self.kappa*d_k - self.nu*dd_k <= f_min*(y_r-1)]

        cons += [-sc[jj,kk] <= 0]
        cons += [sc[jj,kk] <= f_max*y_l]
        cons += [sc[jj,kk] <= f_max*y_r]

    # LQR cost
    lqr_cost = 0.
    for kk in range(self.N):
      lqr_cost += cp.quad_form(x[:,kk]-xg, self.Q)
    for kk in range(self.N-1):
      lqr_cost += cp.quad_form(u[:,kk],self.R)

    self.mlopt_prob = cp.Problem(cp.Minimize(lqr_cost), cons)
    return x, u, y

  # def solve_prob(self, prob_idx, solver=cp.MOSEK):
  #   params = {'x0': self.X0[:,prob_idx], 'xg': self.Xg[:,prob_idx]}
  #   self.solve(params, solver=solver)

  def solve_bin_prob(self, prob_idx, solver=cp.MOSEK):
    params = {'x0': self.X0[:,prob_idx], 'xg': self.Xg[:,prob_idx]}
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
      'y':np.reshape(y_guess, self.Y[:,:,0].shape)}
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
    violations = []
    x = self.X[:,:,prob_idx]
    y = self.Y[:,:,prob_idx]
    sc = self.U[1:,:,prob_idx]

    for kk in range(self.N-1):
      for jj in range(2):
        # Check for when Eq. (27) is strict equality
        if jj == 0:
          d_k    = -x[0,kk] + self.l*x[1,kk] - self.dist
          dd_k   = -x[2,kk] + self.l*x[3,kk]
        else:
          d_k    =  x[0,kk] - self.l*x[1,kk] - self.dist
          dd_k   =  x[2,kk] - self.l*x[3,kk]
        if abs(sc[jj,kk]-self.kappa*d_k-self.nu*dd_k) <= eq_tol:
          violations.append(4*kk + 2*jj)
          violations.append(4*kk + 2*jj + 1)

    return violations

  def construct_features(self, prob_idx): 
    feature_vec = np.array([])
    x0, xg = self.X0[:,prob_idx], self.Xg[:,prob_idx]

    for feature in self.prob_features:
      if feature == "X0":
        feature_vec = np.hstack((feature_vec, x0))
      elif feature == "Xg":
        feature_vec = np.hstack((feature_vec, xg))
      elif feature == "delta2_0":
        d_0 = -x0[0] + self.l*x0[1] - self.dist
        feature_vec = np.hstack((feature_vec, d_0))
      elif feature == "delta3_0":
        d_0 = x0[0] - self.l*x0[1] - self.dist
        feature_vec = np.hstack((feature_vec, d_0))
      elif feature == "delta2_g":
        d_g = -xg[0] + self.l*xg[1] - self.dist
        feature_vec = np.hstack((feature_vec, d_g))
      elif feature == "delta3_g":
        d_g = xg[0] - self.l*xg[1] - self.dist
        feature_vec = np.hstack((feature_vec, d_g))
      elif feature == "dist_to_goal":
        feature_vec = np.hstack((feature_vec, np.linalg.norm(x0-xg)))
      else:
        print('Feature {} is unknown'.format(feature))
    return feature_vec

  def construct_strategies(self):
    n_y = self.Y[:,:,0].size
    self.strategy_dict = {}
    self.training_labels = {}

    n_training_probs = int(round(self.training_batch_percentage * self.n_probs))
    self.features = np.zeros((n_training_probs, self.n_features))
    self.labels = np.zeros((n_training_probs, 1+n_y))
    self.n_strategies = 0

    for ii in range(n_training_probs):
      y_true = np.reshape(self.Y[:,:,ii].T, (n_y))
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

    self.training_params['training_loss'] = {}

    ff_shape.append(self.n_strategies)
    self.training_params['training_loss']['classifier'] = torch.nn.CrossEntropyLoss()
    self.model_classifier = FFNet(ff_shape, activation=torch.nn.ReLU()).cuda()

    ff_shape.pop()
    n_y = self.Y[:,:,0].size
    ff_shape.append(n_y)

    # See: https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/45
    self.training_params['training_loss']['regressor'] = torch.nn.BCEWithLogitsLoss()

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
    training_loss = self.training_params['training_loss']['regressor']

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
    training_loss = self.training_params['training_loss']['classifier']

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

  def solve_with_classifier(self, prob_idx):
    features = self.construct_features(prob_idx)
    input = Variable(torch.from_numpy(features)).float().cuda()
    scores = self.model_classifier(input).cpu().detach().numpy()[:]
    ind_max = np.argsort(scores)[-self.n_evals:][::-1]

    n_y = self.Y[:,:,0].size
    y_guesses = np.zeros((self.n_evals, n_y), dtype=int)

    for k,v in self.training_labels.items():
      for ii,idx in enumerate(ind_max):
        if v[0] == idx:
          y_guesses[ii] = v[1:]

    found_soln = False
    for ii,idx in enumerate(ind_max):
      y_guess = y_guesses[ii]
      prob_success, cost = self.solve_mlopt_prob(prob_idx, y_guess, solver=cp.MOSEK)
      if prob_success:
        print("Succeeded!")
        found_soln = True
        break
    if not found_soln:
      print("Failed!")
