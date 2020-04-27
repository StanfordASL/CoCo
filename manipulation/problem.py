import os
import sys

sys.path.insert(1, os.environ['MLOPT'])
sys.path.insert(1, os.path.join(os.environ['MLOPT'], 'pytorch'))

from optimizer import Optimizer
from models import FFNet, BnBCNN

import pdb
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

system = "manipulation"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Manipulation(Optimizer):
  def __init__(self, n_files=20):
    self.training_batch_percentage = 0.9

    self.solve_time = np.array([], dtype=float) 
    self.J = np.array([], dtype=float) 
    self.node_count = np.array([], dtype=int) 

    n_probs_list = []
    self.n_probs = 0
    for ii in range(n_files):
      fn = os.path.join(os.environ["MLOPT"], system, 'data/testdata{}.h5'.format(1))
      f = h5py.File(fn, 'r')
      if ii == 0:
        self.N_v = int(f['N_v'][()]) 
        self.N_h = int(f['N_h'][()])
        n_y = self.N_v * self.N_h

        self.a = np.empty((12,0), dtype=float)
        # self.f = np.empty((3, 12,n_y,0), dtype=float)
        self.f = []
        self.Y = np.empty((n_y,0), dtype=int)

        self.N_v = f['N_v'][()]
        self.N_h = f['N_h'][()]
        self.num_grasps = f['num_grasps'][()]
        self.mu_min = f['mu_min'][()]
        self.mu_max = f['mu_max'][()]
        self.h_min = f['h_min'][()]
        self.h_max = f['h_max'][()]
        self.r_min = f['r_min'][()]
        self.r_max = f['r_max'][()]
        self.w_std = f['w_std'][()]

      self.a = np.hstack((self.a, f['a'][()].T))
      self.Y = np.hstack((self.Y, f['Y'][()].T))
      f_local = f['f'][()].T
      self.f.append(f_local)
      self.n_probs += f_local.shape[-1]
      n_probs_list.append(f_local.shape[-1])

      self.solve_time = np.append(self.solve_time, f['solve_time'][()])
      self.J = np.append(self.J, f['J'][()])
      self.node_count = np.append(self.node_count, f['node_count'][()])
      f.close()

    f = np.empty((3,12,n_y,self.n_probs))
    ct = 0
    for ii in range(n_files):
      for jj in range(n_probs_list[ii]):
        f[:,:,:,ct] = self.f[ii][:,:,:,jj]
        ct += 1 
    self.f = f

  def construct_bin_prob(self):
    # Saves cvxpy problem object with binary integer values
    # as self.bin_prob
    return    # TODO(pculbertson)

    cons = []

    # cvxpy problem variables
    y = cp.Variable((1), boolean=True)

    # cvxpy problem parameters
    self.bin_prob_parameters = {}    # save list of param names

    cost = 0.
    self.bin_prob = cp.Problem(cp.Minimize(cost), cons) 
    return x, u, y

  def construct_mlopt_prob(self):
    # Saves cvxpy problem object with binary variables as parameters
    # as self.mlopt_prob (otherwise identical to self.bin_prob)
    return    # TODO(pculbertson)

    cons = []

    # cvxpy problem variables

    # cvxpy problem parameters
    y = cp.Parameter((1))
    self.mlopt_prob_parameters = {}    # save list of param names

    cost = 0.
    self.bin_prob = cp.Problem(cp.Minimize(cost), cons) 
    return x, u, y

  def solve_bin_prob_with_idx(self, prob_idx, solver=cp.MOSEK):
    # Construct params for binary problem prob_idx and solve
    return    # TODO(pculbertson)
    params = {} # Dictionary of key/value pairs corresponding to self.bin_prob_parameters
    return self.solve_bin_prob_with_params(params)

  def solve_mlopt_prob_with_idx(self, prob_idx, y_guess, solver=cp.MOSEK):
    # Construct params for mlopt problem prob_idx and solve
    return    # TODO(pculbertson)
    params = {} # Dictionary of key/value pairs corresponding to self.bin_prob_parameters
    return self.solve_mlopt_prob_with_params(params, y_guess)

  def which_M(self, prob_idx, eq_tol=1e-5, ineq_tol=1e-5):
    # Returns list of active logical constraints
    return    # TODO(pculbertson)

    violations = []
    return violations

  def construct_features(self, prob_idx): 
    # Constructs problem features for problem index 
    return    # TODO(pculbertson)
    feature_vec = np.array([])
    for feature in self.prob_features:
      print("Construct feature vector")

    return feature_vec
