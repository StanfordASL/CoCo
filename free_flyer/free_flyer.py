import os
import cvxpy as cp
import pickle
import numpy as np
import pdb
import sys

sys.path.insert(1, os.environ['MLOPT'])

from core import Problem

class FreeFlyer(Problem):
    """Class to setup + solve free-flyer problems."""

    def __init__(self, config=None, solver=cp.MOSEK):
        """Constructor for FreeFlyer class.

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
        self.n = 2; self.m = 2

        self.N, self.Ak, self.Bk, self.Q, self.R, self.n_obs, \
          self.posmin, self.posmax, self.velmin, self.velmax, \
          self.umin, self.umax = prob_params

        self.init_bin_problem()
        self.init_mlopt_problem()

    def init_bin_problem(self):
        cons = []

        # Variables
        x = cp.Variable((2*self.n,self.N)) # state
        u = cp.Variable((self.m,self.N-1))  # control
        y = cp.Variable((4*self.n_obs,self.N-1), boolean=True)
        self.bin_prob_variables = {'x':x, 'u':u, 'y':y}

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

    def init_mlopt_problem(self):
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

    def solve_micp(self, params, solver=cp.MOSEK):
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

    def solve_pinned(self, params, strat, solver=cp.MOSEK):
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

        return prob_success, cost, solve_time

    def which_M(self, x, obstacles, eq_tol=1e-5, ineq_tol=1e-5):
        """Method to check which big-M constraints are active.
        
        Args:
            x: numpy array of size [self.n, self.N], state trajectory.
            obstacles: numpy array of size [4, self.n_obs]
            eq_tol: tolerance for equality constraints, default of 1e-5.
            ineq_tol : tolerance for ineq. constraints, default of 1e-5.
            
        Returns:
            violations: list of which logical constraints are violated.
        """
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

    def construct_features(self, params, prob_features):
        """Helper function to construct feature vector from parameter vector.
        
        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            prob_features: list of strings, desired features for classifier.
        """
        feature_vec = np.array([])

        ## TODO(pculbertson): make this not hardcoded
        x0, xg = params['x0'], params['xg'] 
        obstacles = params['obstacles']

        for feature in prob_features:
          if feature == "X0":
            feature_vec = np.hstack((feature_vec, x0))
          elif feature == "obstacles":
            feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4*self.n_obs))))
          elif feature == "obstacles_map":
            print("obstacles_map feature not implemented yet!")
            continue
            W_H_ratio = posmax[0] / posmax[1]
            H = 32
            W = int(W_H_ratio * H)

            table_img = np.ones((3,H,W))

            # Draw in obstacle of interest last
            obs_ll = [ii for ii in range(n_obs) if ii is not obs_ii]
            obs_ll.append(obs_ii)

            for ll in obs_ll:
                obs = obstacles[ll]
                row_range = range(int(float(obs[2])/posmax[1]*H), int(float(obs[3])/posmax[1]*H))
                col_range = range(int(float(obs[0])/posmax[0]*W), int(float(obs[1])/posmax[0]*W))
                row_range = range(np.maximum(row_range[0],0), np.minimum(row_range[-1],H))
                col_range = range(np.maximum(col_range[0],0), np.minimum(col_range[-1],W))

                if ll is obs_ii:
                    table_img[:, row_range[0]:row_range[-1], col_range[0]:col_range[-1]] = 1
                    table_img[:2, row_range[0]:row_range[-1], col_range[0]:col_range[-1]] = 0
                else:
                    table_img[1:, row_range[0]:row_range[-1], col_range[0]:col_range[-1]] = 0
          else:
            print('Feature {} is unknown'.format(feature))
        return feature_vec
