import os
import cvxpy as cp
import pickle
import numpy as np
import pdb
import sys

sys.path.insert(1, os.environ['MLOPT'])
sys.path.insert(1, os.path.join(os.environ['MLOPT'], 'pytorch'))

from core import Problem, Solver 

class Regression(Solver):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

        # training parameters
        self.training_params = {}
        self.training_params['TRAINING_ITERATIONS'] = int(1500)
        self.training_params['BATCH_SIZE'] = 128
        self.training_params['CHECKPOINT_AFTER'] = int(1000)
        self.training_params['SAVEPOINT_AFTER'] = int(30000)
        self.training_params['TEST_BATCH_SIZE'] = 320

    def forward(self, params):
        pass
