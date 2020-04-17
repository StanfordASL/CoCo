import pdb
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

    self.N = f['N'][()]
    self.n = int(self.X.shape[0])
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

  def construct_problem(self,prob_idx):
    cons = []

    x = cp.Variable((self.n,self.N))
    u = cp.Variable((self.m, self.N-1))
    sc = u[1:,:]
    y = cp.Variable((4, self.N-1), boolean=True)

    # Initial condition
    cons += [x[:,0] == self.X0[:,prob_idx]]

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
      lqr_cost += cp.quad_form(x[:,kk]-self.Xg[:,prob_idx], self.Q)
    for kk in range(self.N-1):
      lqr_cost += cp.quad_form(u[:,kk],self.R)

    self.prob = cp.Problem(cp.Minimize(lqr_cost), cons)
    return x, u, y

  def solve(self):
    self.prob.solve(solver=cp.MOSEK)
