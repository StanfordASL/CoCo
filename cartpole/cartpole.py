import os
import cvxpy as cp
import pickle
import numpy as np

from core import Problem

class Cartpole(Problem):
    """Class to setup + solve cartpole problems."""
    
    def __init__(self, solver=cp.MOSEK):
        """Constructor for Cartpole class.
        
        Args:
            solver: solver object to be used by cvxpy
        """
        super().__init__()
        self.init_problem()
        
    def init_problem(self):
        #setup problem params
        self.n = 4; self.m = 3 
        
        ##TODO(pculbert): allow different sets of params to vary.
        
        relative_path = os.path.dirname(os.path.abspath(__file__))
        infile = open(relative_path+"/cartpole_params.p","rb")
        self.N, self.Ak, self.Bk, self.Q, self.R, self.x_min, self.x_max, \
            self.uc_min, self.uc_max, self.sc_min, self.sc_max, \
            self.delta_min, self.delta_max, self.ddelta_min, self.ddelta_max, \
            self.dh, self.g, self.l, self.mc, self.mp, self.kappa, \
            self.nu, self.dist = pickle.load(infile)
        
        infile.close()
        
        self.init_bin_problem()
        self.init_mlopt_problem()
        
    def init_bin_problem(self):
        
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
            cons += [x[:,kk+1] - (self.Ak @ x[:,kk] 
                + self.Bk @ u[:,kk]) == np.zeros(self.n)]

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
                cons += [self.nu*dd_max*(y_l-1) <= 
                         sc[jj,kk] - self.kappa*d_k - self.nu*dd_k]
                cons += [sc[jj,kk] - self.kappa*d_k - self.nu*dd_k <= 
                         f_min*(y_r-1)]

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

    def init_mlopt_problem(self):
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
        
    def solve_micp(self, params, solver=cp.MOSEK):
        """High-level method to solve parameterized MICP.
        
        Args:
            params: list of numpy arrays, [x0, xg], which are the initial &
                goal state for the current problem.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        #set cvxpy parameters to their values
        x0, xg = params
        self.bin_prob_parameters['x0'] = x0; self.bin_prob_parameters['xg']= xg
        
        #solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        self.bin_prob.solve(solver=solver)
        
        solve_time = self.bin_prob.solver_stats.solve_time
        if self.bin_prob.status == 'optimal':
            prob_success = True
            cost = self.mlopt_prob.value
            
        return prob_success, cost, solve_time
        
    def solve_pinned(self, params, strat, solver=cp.MOSEK):
        raise NotImplementedError
        