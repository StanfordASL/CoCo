import mosek
import cvxpy as cp
import numpy as np

import sys
import pdb
import h5py

class Optimizer():
  def __init__(self, idx):
    fn = 'data/testdata{}.h5'.format(idx+1)
    f = h5py.File(fn, 'r')

    self.X = f['X'][()].T
    self.U = f['U'][()].T
    self.Y = f['Y'][()].T
    self.O = f['O'][()].T

    self.N = f['N'][()]
    self.n_obs = self.O.shape[1]
    self.n = int(self.X.shape[0]/2)
    self.m = int(self.U.shape[0])
    self.n_probs = self.X.shape[-1]

    self.X0 = f['X0'][()].T
    self.Xg = f['Xg'][()].T
    self.solve_time = f['solve_time'][()]
    self.J = f['J'][()]
    self.node_count = f['node_count'][()]

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

  def construct_problem(self,prob_idx):
    cons = []
    obstacles = self.O[:,:,prob_idx]

    # Variables
    x = cp.Variable((2*self.n,self.N)) # state
    u = cp.Variable((self.m,self.N-1))  # control

    # Logic variables
    y = cp.Variable((4*self.n_obs,self.N-1), boolean=True) # binary variables, with no initial constraints on integrality

    cons += [x[:,0] == self.X0[:,prob_idx]]

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
        cons += [sum([y[ii,i_t] for ii in range(yvar_min,yvar_max)])]

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
      lqr_cost += cp.quad_form(x[:,kk]-self.Xg[:,prob_idx], self.Q)
    
    for kk in range(self.N-1):
      lqr_cost += cp.quad_form(u[:,kk], self.R)

    self.prob = cp.Problem(cp.Minimize(lqr_cost), cons)
    return x, u, y
    
  def solve(self):
    self.prob.solve(solver=cp.MOSEK)
