import os
import cvxpy as cp
import pickle
import numpy as np

# ugly path hack :\
import sys
sys.path.append('..')

from core import Problem

class Manipulation(Problem):
    """Class to setup + solve manipulation problems."""

    def __init__(self):
        """Constructor for Manipulation class.

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

    def init_problem(self): 
        self.init_bin_problem()
        self.init_mlopt_problem()

    def init_bin_problem(self):
        return None

    def init_mlopt_problem(self):
        return None

    def solve_micp(self, params, solver=cp.GUROBI):
        return None

    def solve_pinned(self,params,strat):
        return None
