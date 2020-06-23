import os
import cvxpy as cp
import pickle
import numpy as np
import sys
import pdb

sys.path.insert(1, os.environ['MLOPT'])

from core import Problem

class Cartpole(Problem):
    """Class to setup + solve cartpole problems."""

    def __init__(self, config=None, solver=cp.GUROBI):
        """Constructor for Cartpole class.

        Args:
            config: full path to config file. if None, load default config.
            solver: solver object to be used by cvxpy
        """
        super().__init__()

        ## TODO(pculbertson): allow different sets of params to vary.
        if config is None: #use default config
            relative_path = os.path.dirname(os.path.abspath(__file__))
            config = relative_path + '/config/default.p'

        config_file = open(config,"rb")
        _, prob_params, self.sampled_params = pickle.load(config_file)
        config_file.close()
        self.init_problem(prob_params)

    def init_problem(self,prob_params):
        # setup problem params
        self.n = 4; self.m = 3 

        self.N, self.Ak, self.Bk, self.Q, self.R, self.x_min, self.x_max, \
            self.uc_min, self.uc_max, self.sc_min, self.sc_max, \
            self.delta_min, self.delta_max, self.ddelta_min, self.ddelta_max, \
            self.dh, self.g, self.l, self.mc, self.mp, self.kappa, \
            self.nu, self.dist = prob_params

        self.init_bin_problem()
        self.init_mlopt_problem()

    def init_bin_problem(self):
        cons = []

        x = cp.Variable((self.n,self.N))
        u = cp.Variable((self.m, self.N-1))
        sc = u[1:,:]
        y = cp.Variable((4, self.N-1), boolean=True)
        self.bin_prob_variables = {'x': x, 'u' : u, 'y' : y}

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
            cons += [u[0,kk] - self.uc_max <= 0.]

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
            cons += [x[:,kk+1] - (self.Ak @ x[:,kk]
                + self.Bk @ u[:,kk]) == np.zeros(self.n)]

        # State and control constraints
        for kk in range(self.N):
            cons += [self.x_min - x[:,kk] <= np.zeros(self.n)]
            cons += [x[:,kk] - self.x_max <= np.zeros(self.n)]

        for kk in range(self.N-1):
            cons += [self.uc_min - u[0,kk] <= 0.]
            cons += [u[0,kk] - self.uc_max <= 0.]

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

        self.mlopt_prob = cp.Problem(cp.Minimize(lqr_cost), cons)

    def solve_micp(self, params, solver=cp.GUROBI):
        """High-level method to solve parameterized MICP.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy parameters to their values
        for p in self.sampled_params:
            self.bin_prob_parameters[p].value = params[p]
        
        ## TODO(pculbertson): allow different sets of params to vary.
        
        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        x_star, u_star, y_star = None, None, None
        self.bin_prob.solve(solver=solver)
        
        solve_time = self.bin_prob.solver_stats.solve_time
        if self.bin_prob.status == 'optimal':
            prob_success = True
            cost = self.bin_prob.value
            x_star = self.bin_prob_variables['x'].value
            u_star = self.bin_prob_variables['u'].value
            y_star = self.bin_prob_variables['y'].value.astype(int)
            
        return prob_success, cost, solve_time, (x_star, u_star, y_star)
        
    def solve_pinned(self, params, strat, solver=cp.GUROBI):
        """High-level method to solve MICP with pinned params & integer values.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            strat: numpy integer array, corresponding to integer values for the
                desired strategy.
            solver: cvxpy Solver object; defaults to Mosek.
        """
        # set cvxpy params to their values
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = params[p]

        self.mlopt_prob_parameters['y'].value = strat

        ## TODO(pculbertson): allow different sets of params to vary.

        # solve problem with cvxpy
        prob_success, cost, solve_time = False, np.Inf, np.Inf
        self.mlopt_prob.solve(solver=solver)

        solve_time = self.mlopt_prob.solver_stats.solve_time
        if self.mlopt_prob.status == 'optimal':
            prob_success = True
            cost = self.mlopt_prob.value

        # Clear any saved params
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = None
        self.mlopt_prob_parameters['y'].value = None

        return prob_success, cost, solve_time

    def which_M(self, x, u, eq_tol=1e-5, ineq_tol=1e-5):
        """Method to check which big-M constraints are active.
        
        Args:
            x: numpy array of size [self.n, self.N], state trajectory.
            u: numpy array of size [self.m, self.N], input trajectory.
            eq_tol: tolerance for equality constraints, default of 1e-5.
            ineq_tol : tolerance for ineq. constraints, default of 1e-5.
            
        Returns:
            violations: list of which logical constraints are violated.
        """

        violations = []
        sc = u[1:,:]

        for kk in range(self.N-1):
            for jj in range(2):
                # Check for when Eq. (27) is strict equality
                if jj == 0:
                    d_k = -x[0,kk] + self.l*x[1,kk] - self.dist
                    dd_k = -x[2,kk] + self.l*x[3,kk]
                else:
                    d_k = x[0,kk] - self.l*x[1,kk] - self.dist
                    dd_k = x[2,kk] - self.l*x[3,kk]
                if abs(sc[jj,kk]-self.kappa*d_k-self.nu*dd_k) <= eq_tol:
                    violations.append(4*kk + 2*jj)
                    violations.append(4*kk + 2*jj + 1)

        return violations

    def construct_features(self, params, prob_features):
        """Helper function to construct feature vector from parameter vector.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            prob_features: list of strings, desired features for classifier.
        """
        feature_vec = np.array([])
        x0, xg = params['x0'], params['xg'] 
        ## TODO(pculbertson): make this not hardcoded

        for feature in prob_features:
            if feature == "x0":
                feature_vec = np.hstack((feature_vec, x0))
            elif feature == "xg":
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
