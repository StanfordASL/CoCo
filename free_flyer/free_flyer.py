import os
import cvxpy as cp
import yaml
import pickle
import numpy as np
import sys
import pdb

sys.path.insert(1, os.environ['CoCo'])

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

        self.H = 64
        self.W = int(self.posmax[0] / self.posmax[1] * self.H)
        self.H, self.W = 32, 32

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
            for jj in range(self.m):
                cons += [self.umin - u[jj,kk] <= 0]
                cons += [u[jj,kk] - self.umax <= 0]

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
        self.mlopt_prob_variables = {'x':x, 'u':u}

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
            for jj in range(self.m):
                cons += [self.umin - u[jj,kk] <= 0]
                cons += [u[jj,kk] - self.umax <= 0]

        M = 1000. # big M value
        lqr_cost = 0.
        # l2-norm of lqr_cost
        for kk in range(self.N):
          lqr_cost += cp.quad_form(x[:,kk]-xg, self.Q)

        for kk in range(self.N-1):
          lqr_cost += cp.quad_form(u[:,kk], self.R)

        self.mlopt_prob = cp.Problem(cp.Minimize(lqr_cost), cons)

    def solve_micp(self, params, solver=cp.MOSEK, msk_param_dict=None):
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
        if solver == cp.MOSEK:
            # See: https://docs.mosek.com/9.1/dotnetfusion/param-groups.html#doc-param-groups
            if not msk_param_dict:
              msk_param_dict = {}
              with open(os.path.join(os.environ['CoCo'], 'config/mosek.yaml')) as file:
                  msk_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.bin_prob.solve(solver=solver, mosek_params=msk_param_dict)
        elif solver == cp.GUROBI:
            grb_param_dict = {}
            with open(os.path.join(os.environ['CoCo'], 'config/gurobi.yaml')) as file:
                grb_param_dict = yaml.load(file, Loader=yaml.FullLoader)

            self.bin_prob.solve(solver=solver, **grb_param_dict)
        solve_time = self.bin_prob.solver_stats.solve_time

        x_star, u_star, y_star = None, None, None
        if self.bin_prob.status in ['optimal', 'optimal_inaccurate'] and self.bin_prob.status not in ['infeasible', 'unbounded']:
            prob_success = True
            cost = self.bin_prob.value
            x_star = self.bin_prob_variables['x'].value
            u_star = self.bin_prob_variables['u'].value
            y_star = self.bin_prob_variables['y'].value.astype(int)

        # Clear any saved params
        for p in self.sampled_params:
            self.bin_prob_parameters[p].value = None

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
        x_star, u_star, y_star = None, None, strat
        if self.mlopt_prob.status == 'optimal':
            prob_success = True
            cost = self.mlopt_prob.value
            x_star = self.mlopt_prob_variables['x'].value
            u_star = self.mlopt_prob_variables['u'].value

        # Clear any saved params
        for p in self.sampled_params:
            self.mlopt_prob_parameters[p].value = None
        self.mlopt_prob_parameters['y'].value = None

        return prob_success, cost, solve_time, (x_star, u_star, y_star)

    def which_M(self, x, obstacles, eq_tol=1e-5, ineq_tol=1e-5):
        """Method to check which big-M constraints are active.
        
        Args:
            x: numpy array of size [2*self.n, self.N], state trajectory.
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
                    if (x[i_dim,i_t+1] - o_min > ineq_tol):
                        curr_violations.append(self.n*i_dim + 2*self.n*i_t)

                    o_max = obstacles[self.n*i_dim+1,i_obs]
                    if (-x[i_dim,i_t+1]  + o_max > ineq_tol):
                        curr_violations.append(self.n*i_dim+1 + 2*self.n*i_t)
            curr_violations = list(set(curr_violations))
            curr_violations.sort()
            violations.append(curr_violations)
        return violations

    def construct_features(self, params, prob_features, ii_obs=None):
        """Helper function to construct feature vector from parameter vector.

        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            prob_features: list of strings, desired features for classifier.
            ii_obs: index of obstacle strategy being queried; appends one-hot
                encoding to end of feature vector
        """
        feature_vec = np.array([])

        ## TODO(pculbertson): make this not hardcoded
        x0, xg = params['x0'], params['xg'] 
        obstacles = params['obstacles']

        for feature in prob_features:
            if feature == "x0":
                feature_vec = np.hstack((feature_vec, x0))
            elif feature == "xg":
                feature_vec = np.hstack((feature_vec, xg))
            elif feature == "obstacles":
                feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4*self.n_obs))))
            elif feature == "obstacles_map":
                continue
            else:
                print('Feature {} is unknown'.format(feature))

        # Append one-hot encoding to end
        if ii_obs is not None:
            one_hot = np.zeros(self.n_obs)
            one_hot[ii_obs] = 1.
            feature_vec = np.hstack((feature_vec, one_hot))

        return feature_vec

    def construct_cnn_features(self, params, prob_features, ii_obs=None):
        """Helper function to construct 3xHxW image for CNN with
                obstacles shaded in blue and ii_obs shaded in red

        Args:
            params: Dict of param values; keys are self.sampled_params,
                values are numpy arrays of specific param values.
            prob_features: list of strings, desired features for classifier.
            ii_obs: index of obstacle strategy being queried; appends one-hot
                encoding to end of feature vector
        """
        if "obstacles_map" not in prob_features:
            return None

        obstacles = params['obstacles']

        # W_H_ratio = self.posmax[0] / self.posmax[1]
        # H = 32
        # W = int(W_H_ratio * H)
        H, W = self.H, self.W

        posmin, posmax = self.posmin, self.posmax

        table_img = np.ones((3,H,W))

        # If a particular obstacle requested, shade that in last
        obs_list = [ii for ii in range(self.n_obs) if ii is not ii_obs]
        if ii_obs is not None:
            obs_list.append(ii_obs)

        for ll in obs_list:
            obs = obstacles[:,ll]
            row_range = range(int(float(obs[2])/posmax[1]*H), int(float(obs[3])/posmax[1]*H))
            col_range = range(int(float(obs[0])/posmax[0]*W), int(float(obs[1])/posmax[0]*W))
            row_range = range(np.maximum(row_range[0],0), np.minimum(row_range[-1],H))
            col_range = range(np.maximum(col_range[0],0), np.minimum(col_range[-1],W))

            # 0 out RG channels, leaving only B channel on
            table_img[:2, row_range[0]:row_range[-1], col_range[0]:col_range[-1]] = 0.

            if ii_obs is not None and ll is ii_obs:
                # 0 out all channels and then turn R channel on
                table_img[:, row_range[0]:row_range[-1], col_range[0]:col_range[-1]] = 0.
                table_img[0, row_range[0]:row_range[-1], col_range[0]:col_range[-1]] = 1.

        return table_img

    def propagate_features(self, feature_vec, u0, prob_features):
        if "x0" in prob_features:
            x0 = feature_vec[:2*self.n]
            feature_vec[:2*self.n] = self.Ak @ x0 + self.Bk @ u0
        return feature_vec
