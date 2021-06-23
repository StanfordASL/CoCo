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

        self.reset_gusto_params()
        self.init_gusto_problem()

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

    def init_gusto_problem(self):
        cons = []

        # Variables
        x = cp.Variable((2*self.n,self.N)) # state
        u = cp.Variable((self.m,self.N-1))  # control
        xlb_slack_vars = cp.Variable((2*self.n, self.N), nonneg=True)
        xub_slack_vars = cp.Variable((2*self.n, self.N), nonneg=True)
        tr_slack_vars = cp.Variable(2*self.N-1, nonneg=True)
        sdf_slack_vars = cp.Variable(self.N*self.n_obs, nonneg=True)
        self.gusto_prob_variables = {'x':x, 'u':u,
          'xlb_slack_vars':xlb_slack_vars, 'xub_slack_vars':xub_slack_vars,
          'tr_slack_vars':tr_slack_vars, 'sdf_slack_vars':sdf_slack_vars}

        # Parameters
        x0 = cp.Parameter(2*self.n)
        xg = cp.Parameter(2*self.n)
        obstacles = cp.Parameter((4, self.n_obs))
        omega = cp.Parameter(1)
        delta = cp.Parameter(1)
        x_bar = cp.Parameter((2*self.n, self.N)) # state reference
        u_bar = cp.Parameter((self.m, self.N-1))  # control reference
        dists = cp.Parameter(self.N*self.n_obs)
        sdf_grads = cp.Parameter((self.n, self.N*self.n_obs))
        self.gusto_prob_parameters = {'x0': x0, 'xg': xg, 'obstacles': obstacles,
          'omega':omega, 'delta':delta, 'x_bar':x_bar, 'u_bar':u_bar,
          'dists':dists, 'sdf_grads':sdf_grads}

        # Initial condition
        cons += [x[:,0] == x0]

        # Dynamics constraints
        for ii in range(self.N-1):
            cons += [x[:,ii+1] - (self.Ak @ x[:,ii] + self.Bk @ u[:,ii]) == np.zeros(2*self.n)]

        # Control constraints
        for i_t in range(self.N-1):
            cons += [cp.norm(u[:, i_t]) <= self.umax]

        penalty_cost = 0.0
        # State bounds penalties
        for i_t in range(self.N):
            for jj in range(self.n):
                cons += [self.posmin[jj] - x[jj,i_t] <= xlb_slack_vars[jj, i_t]]
                cons += [x[jj,i_t] - self.posmax[jj] <= xub_slack_vars[jj, i_t]]
                cons += [self.velmin - x[self.n+jj, i_t] <= xlb_slack_vars[self.n+jj, i_t]]
                cons += [x[self.n+jj, i_t] - self.velmax <= xub_slack_vars[self.n+jj, i_t]]
        penalty_cost += omega*cp.sum(xlb_slack_vars) + omega*cp.sum(xub_slack_vars)

        # Collision avoidance penalty
        for i_t in range(self.N):
            for ii_obs in range(self.n_obs):
                n_hat, dist = sdf_grads[:, self.n_obs*i_t+ii_obs], dists[self.n_obs*i_t+ii_obs]
                cons += [-(dist + n_hat.T @ (x[:self.n, i_t] - x_bar[:self.n, i_t])) <= sdf_slack_vars[self.n_obs*i_t+ii_obs]]
        penalty_cost += omega*cp.sum(sdf_slack_vars)

        # Trust region penalties
        for i_t in range(self.N):
            cons += [cp.norm(x[:, i_t] - x_bar[:, i_t], 1) - delta <= tr_slack_vars[i_t]]
        for i_t in range(self.N-1):
            cons += [cp.norm(u[:, i_t] - u_bar[:, i_t], 1) - delta <= tr_slack_vars[self.N+i_t]]
        penalty_cost += omega*cp.sum(tr_slack_vars)

        lqr_cost = 0.
        # l2-norm of lqr_cost
        for i_t in range(self.N):
            lqr_cost += cp.quad_form(x[:, i_t]-xg, self.Q)

        for i_t in range(self.N-1):
            lqr_cost += cp.quad_form(u[:, i_t], self.R)

        self.gusto_prob = cp.Problem(cp.Minimize(lqr_cost+penalty_cost), cons)

    def solve_gusto_problem(self, params, max_iter, solver=cp.GUROBI):
        x0, xg, obstacles = params['x0'], params['xg'], params['obstacles']

        self.reset_gusto_params()
        omega, delta = self.omega0, self.delta0

        gusto_params = {}
        for p in params:
            gusto_params[p] = params[p]

        # Linear interpolation between x0 and xg for state guess
        x_bar, u_bar = np.zeros((2*self.n, self.N)), np.zeros((self.m, self.N-1))
        for ii in range(2*self.n):
            x_bar[ii,:] = np.linspace(x0[ii], xg[ii], self.N)

        if solver == cp.MOSEK:
            # See: https://docs.mosek.com/9.1/dotnetfusion/param-groups.html#doc-param-groups
            if not msk_param_dict:
              msk_param_dict = {}
              with open(os.path.join(os.environ['CoCo'], 'config/mosek.yaml')) as file:
                  msk_param_dict = yaml.load(file, Loader=yaml.FullLoader)
        elif solver == cp.GUROBI:
            grb_param_dict = {}
            with open(os.path.join(os.environ['CoCo'], 'config/gurobi.yaml')) as file:
                grb_param_dict = yaml.load(file, Loader=yaml.FullLoader)

        prob_success, cost, solve_time = False, np.Inf, 0.0
        solve_gusto, ii_iter = True, 0
        while solve_gusto and ii_iter < max_iter:
            ii_iter += 1

            dists = np.zeros(self.N*self.n_obs)
            sdf_grads = np.zeros((self.n, self.N*self.n_obs))
            for i_t in range(self.N):
                for ii_obs in range(self.n_obs):
                    obs = obstacles[:, ii_obs]
                    dist, n_hat = self.get_sdf(x0, obs)
                    dists[self.n_obs*i_t+ii_obs] = dist
                    sdf_grads[:, self.n_obs*i_t+ii_obs] = n_hat

            gusto_params['omega'] = np.array([omega])
            gusto_params['delta'] = np.array([delta])
            gusto_params['x_bar'] = x_bar
            gusto_params['u_bar'] = u_bar
            gusto_params['dists'] = dists
            gusto_params['sdf_grads'] = sdf_grads

            for p in self.gusto_prob_parameters:
                self.gusto_prob_parameters[p].value = gusto_params[p]

            if solver == cp.MOSEK:
                self.gusto_prob.solve(solver=solver, mosek_params=msk_param_dict)
            elif solver == cp.GUROBI:
                self.gusto_prob.solve(solver=solver, **grb_param_dict)
            solve_time += self.gusto_prob.solver_stats.solve_time

            x_star, u_star, y_star = None, None, None
            if self.gusto_prob.status in ['optimal', 'optimal_inaccurate'] and self.gusto_prob.status not in ['infeasible', 'unbounded']:
                prob_success = True
                x_star = self.gusto_prob_variables['x'].value
                u_star = self.gusto_prob_variables['u'].value
            else:
                prob_success, accept_soln, solve_gusto = False, False, False
                continue

            # Check for trust region satisfaction
            trust_region_violated = False
            for i_t in range(self.N):
                if np.linalg.norm(x_star[:, i_t] - x_bar[:, i_t], 1) > delta:
                    trust_region_violated = True
                    break

                if i_t == self.N-1:
                    continue
                if np.linalg.norm(u_star[:, i_t] - u_bar[:, i_t], 1) > delta:
                    trust_region_violated = True
                    break

            if trust_region_violated:
                accept_soln = False
                delta = delta
                omega = self.gamma_fail*omega
            else:
                accept_soln = False
                rho = self.trust_region_ratio(x_star, x_bar, gusto_params)
                print(rho)
                if rho > self.rho1:
                    delta *= self.beta_fail
                    omega = omega
                    accept_soln = False
                else:
                    accept_soln = True
                    if rho < self.rho0:
                        delta *= self.beta_succ
                    else:
                        delta = delta

                cvx_con_violated = False
                for i_t in range(self.N):
                    for jj in range(self.n):
                        if self.posmin[jj] - x_star[jj, i_t] > 0 or \
                          x_star[jj, i_t] - self.posmax[jj] > 0 or \
                          self.velmin - x_star[self.n+jj, i_t] > 0 or \
                          x_star[self.n+jj, i_t] - self.velmax > 0:
                            cvx_con_violated = True
                            break

                if cvx_con_violated:
                    omega = self.gamma_fail * omega
                else:
                    omega = omega

                if accept_soln:
                    primal_diff = 0.0
                    for i_t in range(self.N):
                        primal_diff += np.linalg.norm(x_star[:, i_t] - x_bar[:, i_t], 1)
                        if i_t == self.N-1:
                            continue
                        primal_diff += np.linalg.norm(u_star[:, i_t] - u_bar[:, i_t], 1)
                    if primal_diff < self.gusto_conv_thresh:
                        solve_gusto = False

                    x_bar, u_bar = x_star, u_star

        for key in self.gusto_prob_parameters:
            self.gusto_prob_parameters[key].value = None

        # Check for state constraint satisfaction
        state_cons_satisfied = True
        ineq_tol = 1e-5
        for i_t in range(self.N):
            if not state_cons_satisfied:
                continue

            for jj in range(self.n):
                if self.posmin[jj] - x_star[jj, i_t] > ineq_tol:
                    state_cons_satisfied = False
                if x_star[jj, i_t] - self.posmax[jj] > ineq_tol:
                    state_cons_satisfied = False
                if self.velmin - x_star[self.n+jj, i_t] > ineq_tol:
                    state_cons_satisfied = False
                if x_star[self.n+jj, i_t] - self.velmax > ineq_tol:
                    state_cons_satisfied = False

            for ii_obs in range(self.n_obs):
                if x_star[0, i_t] + ineq_tol >= obstacles[0, ii_obs] and x_star[0, i_t] - ineq_tol <= obstacles[1, ii_obs] and \
                  x_star[1, i_t] + ineq_tol >= obstacles[2, ii_obs] and x_star[1, i_t]  - ineq_tol <= obstacles[3, ii_obs]:
                    state_cons_satisfied = False
        prob_success = accept_soln and state_cons_satisfied

        lqr_cost = np.Inf
        if prob_success:
            # l2-norm of lqr_cost
            lqr_cost = 0.
            for i_t in range(self.N):
                lqr_cost += (x_star[:, i_t] - xg).T @ self.Q @ (x_star[:, i_t] - xg)
            for i_t in range(self.N-1):
                lqr_cost += u_star[:, i_t].T @ self.R @ u_star[:, i_t]

        return prob_success, lqr_cost, solve_time, ii_iter, (x_star, u_star)

    def reset_gusto_params(self):
        # Parameters for GuSTO algorithm
        self.omega0 = 100.0
        self.omega_max = 100000.0
        self.omega_times = 10.0
        self.eps = 1e-6
        self.rho0 = 0.5
        self.rho1 = 0.9
        self.beta_succ = 2.0
        self.beta_fail = 0.5
        self.delta0 = 1000.0
        self.gamma_fail = 5.0
        self.gusto_conv_thresh = 1e-2

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

    def get_sdf(self, x0, obs):
        x, y = x0[:self.n]
        x_min, x_max, y_min, y_max = obs
        sign = 1.0
        if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
            sign = -1.0

        x_proj = np.minimum(np.abs(x-x_min), np.abs(x-x_max))
        y_proj = np.minimum(np.abs(y-y_min), np.abs(y-y_max))
        dist = sign*np.minimum(x_proj, y_proj)

        n_hat = np.array([0.0, 0.0])
        if x_proj < y_proj:
            if np.abs(x-x_min) < np.abs(x-x_max):
                n_hat = np.array([x-x_min, 0.0])
            else:
                n_hat = np.array([x-x_max, 0.0])
        else:
            if np.abs(y-y_min) < np.abs(y-y_max):
                n_hat = np.array([y-y_min, 0.0])
            else:
                n_hat = np.array([y-y_max, 0.0])
        n_hat *= sign
        return dist, n_hat

    def trust_region_ratio(self, x_star, x_bar, params):
        num, den = 0.0, 0.0

        obstacles = params['obstacles']
        for i_t in range(self.N):
              x_it, x_it_bar = x_star[:self.n, i_t], x_bar[:self.n, i_t]
              for ii_obs in range(self.n_obs):
                  obs = obstacles[:, ii_obs]
                  dist, n_hat = self.get_sdf(x_it_bar, obs)
                  n_hat = n_hat / np.linalg.norm(n_hat)
                  linearized = - (dist + n_hat.T @ (x_it - x_it_bar))

                  dist, n_hat = self.get_sdf(x_it, obs)
                  n_hat = n_hat / np.linalg.norm(n_hat)
                  num += np.abs(-dist - linearized)
                  den += np.abs(linearized)
        return num / den
