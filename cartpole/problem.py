import os
import sys

sys.path.insert(1, os.environ['MLOPT'])
sys.path.insert(1, os.path.join(os.environ['MLOPT'], 'pytorch'))

from optimizer import Optimizer
from models import FFNet, BnBCNN

import pdb

import osqp
import mosek
import cvxpy as cp
import numpy as np

import pdb
import h5py

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

system = "cartpole"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Cartpole(Optimizer):
  def __init__(self, n_files=20):
    self.n = 4
    self.m = 3 

    self.training_batch_percentage = 0.9

    self.solve_time = np.array([], dtype=float) 
    self.J = np.array([], dtype=float) 
    self.node_count = np.array([], dtype=int) 

    for ii in range(n_files):
      fn = os.path.join(os.environ["MLOPT"], system, 'data/testdata{}.h5'.format(ii+1))
      f = h5py.File(fn, 'r')
      if ii == 0:
        self.N = f['N'][()]

        self.X = np.empty((self.n, self.N,0), dtype=float)
        self.U = np.empty((self.m, self.N-1, 0), dtype=float)
        self.Y = np.empty((4, self.N-1,0), dtype=int)
        self.X0 = np.empty((self.n, 0), dtype=float)
        self.Xg = np.empty((self.n, 0), dtype=float)

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
      
    self.n_y = 4*(self.N-1)
    
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

  def solve_bin_prob_with_idx(self, prob_idx, solver=cp.MOSEK):
    # Construct params for binary problem prob_idx and solve
    params = {'x0': self.X0[:,prob_idx], 'xg': self.Xg[:,prob_idx]}
    return self.solve_bin_prob_with_params(params, solver=solver)

  def solve_mlopt_prob_with_idx(self, prob_idx, y_guess, solver=cp.MOSEK):
    params = {'x0': self.X0[:,prob_idx], 'xg': self.Xg[:,prob_idx],
      'y':np.reshape(y_guess, self.Y[:,:,0].T.shape).T}
    return self.solve_mlopt_prob_with_params(params, y_guess, solver=solver)

  def which_M(self, prob_idx, eq_tol=1e-5, ineq_tol=1e-5):
    # Returns list of active logical constraints
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
