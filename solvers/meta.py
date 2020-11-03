import pdb
import time
import random
import sys
import itertools
import pickle, os
import numpy as np
import cvxpy as cp

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
from datetime import datetime

sys.path.insert(1, os.environ['MLOPT'])
sys.path.insert(1, os.path.join(os.environ['MLOPT'], 'pytorch'))

from core import Problem, Solver
from pytorch.models import FFNet, CNNet
from .mlopt_ff import MLOPT_FF

class Meta(MLOPT_FF):
    """
    Constructor for meta-learning variant of MLOPT_FF
    """
    def __init__(self, system, problem, prob_features, n_evals=2):
        super().__init__(system, problem, prob_features)

        self.update_lr = 1e-3
        self.meta_lr = 1e-3
        self.n_way = 5
        self.k_spt = 1
        self.k_qry = 15
        self.task_num = 4
        self.update_step = 2
        self.update_step_test = 1
        self.margin = 10.

    def construct_strategies(self, n_features, train_data, device_id=1):
        super().construct_strategies(n_features, train_data)
        super().setup_network(device_id=device_id)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'MetaCoCoFF_{}_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(self.system, now)
        feas_fn = 'MetaCoCoFF_feas_{}_{}.pt'
        feas_fn = os.path.join(os.getcwd(), feas_fn)
        self.feas_fn = feas_fn.format(self.system, now)

        self.feas_model = deepcopy(self.model).to(device=self.device)

    def load_network(self, coco_model_fn, feas_model_fn=""):
        super().load_network(coco_model_fn)

        if os.path.exists(feas_model_fn):
            print('Loading presaved feasibility model from {}'.format(feas_model_fn))
            # self.feas_model.load_state_dict(torch.load(feas_model_fn))
            saved_params = list(torch.load(feas_model_fn).values())
            for ii in range(len(saved_params)):
                self.feas_model.vars[ii].data.copy_(saved_params[ii])
            self.feas_fn = feas_model_fn
        else:
            self.copy_shared_params(self.feas_model, self.model)

    def copy_shared_params(self, target, source):
        """
        Copies all network weights from source NN to target NN
        except for the last layer
        """
        target_weights = list(target.parameters())
        source_weights = list(source.parameters())
        for ii in range(len(source_weights)-2):
            target_weights[ii].data.copy_(source_weights[ii].detach())

    def copy_all_but_last(self, target, source_data):
        """
        Updates target NN parameters using list of weights in source_data except for last layer
        """
        target_weights = list(target.parameters())
        for ii in range(len(source_data)-2):
            target_weights[ii].data.copy_(source_data[ii].data)

    def copy_last(self, target, source_data):
        """
        Updates last layer NN parameters in target last layer weights in source_data
        """
        target_weights = list(target.parameters())
        target_weights[-2].data.copy_(source_data[-2].data)
        target_weights[-1].data.copy_(source_data[-1].data)

    def train(self, train_data, summary_writer_fn):
        # grab training params
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        TRAINING_ITERATIONS = self.training_params['TRAINING_ITERATIONS']
        BATCH_SIZE = self.training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = self.training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = self.training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = self.training_params['TEST_BATCH_SIZE']

        NUM_META_PROBLEMS = np.maximum(int(np.ceil(BATCH_SIZE / self.problem.n_obs)), 10)

        model = self.model
        feas_model = self.feas_model
        writer = SummaryWriter("{}".format(summary_writer_fn))

        params = train_data[0]
        X = self.features[:self.problem.n_obs*self.num_train]
        X_cnn = np.zeros((BATCH_SIZE, 3,self.problem.H,self.problem.W))
        Y = self.labels[:self.problem.n_obs*self.num_train,0]

        training_loss = torch.nn.CrossEntropyLoss()
        inner_training_loss = torch.nn.HingeEmbeddingLoss(margin=self.margin, reduction='mean')
        meta_opt = torch.optim.Adam(model.parameters(), lr=self.meta_lr, weight_decay=0.00001)

        def solve_pinned_worker(prob_params, y_guesses, idx, return_dict):
            return_dict[idx] = False
            # Sometimes Mosek fails, so try again with Gurobi
            try:
                return_dict[idx] = self.problem.solve_pinned(prob_params[idx], y_guesses[idx], solver=cp.MOSEK)[0]
            except:
                return_dict[idx] = self.problem.solve_pinned(prob_params[idx], y_guesses[idx], solver=cp.GUROBI)[0]

        itr = 1
        for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(0,X.shape[0]-1))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii,idx in enumerate(indices):
                # fast_weights are network weights for feas_model with descent steps taken
                fast_weights = list(feas_model.parameters())
                for _ in range(self.update_step):
                    inner_loss = torch.zeros(1).to(device=self.device)

                    # Update feas_model to be using fast_weights
                    self.copy_last(feas_model, fast_weights)
                    self.copy_all_but_last(feas_model, fast_weights)

                    idx_vals = np.random.randint(0,self.num_train, NUM_META_PROBLEMS)
                    prob_params_list = []

                    # Compute feature vectors for each of the indices in idx_vals
                    # Note each problem itself has self.problem.n_obs task features
                    ff_inputs_inner = torch.zeros((NUM_META_PROBLEMS*self.problem.n_obs, self.n_features)).to(device=self.device)
                    cnn_inputs_inner = torch.zeros((NUM_META_PROBLEMS*self.problem.n_obs, 3, self.problem.H, self.problem.W)).to(device=self.device)

                    for prb_idx, idx_val in enumerate(idx_vals):
                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][idx_val]
                        prob_params_list.append(prob_params)

                        # Grab strategy indices and feasibility scores for a particular problem
                        prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))

                        ff_inner = torch.from_numpy(self.problem.construct_features(prob_params, self.prob_features))
                        ff_inputs_inner[prb_idx_range] = ff_inner.repeat(self.problem.n_obs,1).float().to(device=self.device)

                        X_cnn_inner = np.zeros((self.problem.n_obs, 3, self.problem.H, self.problem.W))
                        for ii_obs in range(self.problem.n_obs):
                            X_cnn_inner[ii_obs] = self.problem.construct_cnn_features(prob_params, \
                                            self.prob_features, \
                                            ii_obs=ii_obs)
                        cnn_inputs_inner[prb_idx_range] = torch.from_numpy(X_cnn_inner).float().to(device=self.device)

                    # Use strategy classifier to identify high ranking strategies for each feature
                    class_scores = model(cnn_inputs_inner, ff_inputs_inner).detach().cpu().numpy()
                    class_labels = np.argmax(class_scores, axis=1)      # Predicted strategy index for each features of length NUM_META_PROBLEMS*self.problem.n_obs

                    feas_scores = feas_model(cnn_inputs_inner, ff_inputs_inner, vars=fast_weights)

                    y_guesses = -1*np.ones((NUM_META_PROBLEMS, 4*self.problem.n_obs, self.problem.N-1), dtype=int)
                    for prb_idx in range(NUM_META_PROBLEMS):
                        # Grab strategy indices for a particular problem
                        prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))
                        class_labels_prb = class_labels[prb_idx_range]
                        for cl_ii, cl in enumerate(class_labels_prb):
                            cl_idx = np.where(self.labels[:,0] == cl)[0][0]
                            y_obs = self.labels[cl_idx, 1:]
                            y_guesses[prb_idx, 4*cl_ii:4*(cl_ii+1)] = np.reshape(y_obs, (4, self.problem.N-1))

                    # TODO(acauligi): use multiprocessing to parallelize this
                    prob_success_dict = {}
                    for prb_idx, idx_val in enumerate(idx_vals):
                        prob_success_dict[prb_idx] = False
                        try:
                            prob_success_dict[prb_idx] = self.problem.solve_pinned(prob_params_list[prb_idx], y_guesses[prb_idx], solver=cp.MOSEK)[0]
                        except:
                            prob_success_dict[prb_idx] = self.problem.solve_pinned(prob_params_list[prb_idx], y_guesses[prb_idx], solver=cp.GUROBI)[0]

                    # Compute hinge loss using class score from each applied strategy
                    labels, losses = torch.zeros(NUM_META_PROBLEMS*self.problem.n_obs).to(device=self.device), torch.zeros(NUM_META_PROBLEMS*self.problem.n_obs).to(device=self.device)
                    for prb_idx in range(NUM_META_PROBLEMS):
                        # Grab strategy indices and feasibility scores for a particular problem
                        prb_idx_range = range(self.problem.n_obs*prb_idx, self.problem.n_obs*(prb_idx+1))

                        # If solution was feasible, label = 1., if infeasible label = -1.
                        labels[prb_idx_range] = 1. if prob_success_dict[prb_idx] else -1.

                        feas_scores_prb = feas_scores[prb_idx_range]
                        for ii_obs in range(self.problem.n_obs):
                            losses[prb_idx_range[ii_obs]] = feas_scores_prb[ii_obs, class_labels_prb[ii_obs]]

                    inner_loss = inner_training_loss(losses, labels).to(device=self.device)

                    # Descent step on feas_model network weights
                    grad = torch.autograd.grad(inner_loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                ff_inputs = torch.from_numpy(X[idx,:]).float().to(device=self.device)
                labels = torch.from_numpy(Y[idx]).long().to(device=self.device)

                # forward + backward + optimize
                X_cnn = np.zeros((len(idx), 3,self.problem.H,self.problem.W))
                for idx_ii, idx_val in enumerate(idx):
                    prob_params = {}
                    for k in params:
                        prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                    X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                cnn_inputs = torch.from_numpy(X_cnn).float().to(device=self.device)

                # Pass inner loop weights to MLOPT classifier (except last layer)
                # self.copy_all_but_last(model, fast_weights)
                model_weights = list(model.parameters())
                model_weights[:-2] = fast_weights[:-2]

                outputs = model(cnn_inputs, ff_inputs, vars=model_weights)

                loss = training_loss(outputs, labels).float().to(device=self.device)
                running_loss += loss.item()

                class_guesses = torch.argmax(outputs,1)
                accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                loss.backward()
                meta_opt.step()
                meta_opt.zero_grad() # zero the parameter gradients

                # Update feas_model weights
                self.copy_last(feas_model, [fw.detach() for fw in fast_weights])
                self.copy_shared_params(feas_model, model)

                if itr % CHECKPOINT_AFTER == 0:
                    rand_idx = list(np.arange(0,X.shape[0]-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    ff_inputs = torch.from_numpy(X[test_inds,:]).float().to(device=self.device)
                    labels = torch.from_numpy(Y[test_inds]).long().to(device=self.device)

                    # forward + backward + optimize
                    X_cnn = np.zeros((len(test_inds), 3,self.problem.H,self.problem.W))
                    for idx_ii, idx_val in enumerate(test_inds):
                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                        X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                    cnn_inputs = torch.from_numpy(X_cnn).float().to(device=self.device)
                    outputs = model(cnn_inputs, ff_inputs)

                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                    print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    torch.save(feas_model.state_dict(), self.feas_fn)
                    writer.add_scalar('Loss/train', running_loss / float(SAVEPOINT_AFTER), itr)
                    running_loss = 0.

                itr += 1
            if itr % 10 == 0:
                print(itr)

    def finetuning(self, test_data, summary_writer_fn, n_evals=2, max_evals=16):
        """
        Step through test_data and perform inner loop update on self.feas_model
        to fine tune self.model
        """
        params, x_test = test_data[:2]
        n_test = x_test.shape[0]

        model, feas_model = self.model, self.feas_model
        inner_training_loss = torch.nn.HingeEmbeddingLoss(margin=self.margin, reduction='mean')

        BATCH_SIZE = self.training_params['BATCH_SIZE']
        rand_idx = list(np.arange(0,n_test))
        random.shuffle(rand_idx)
        indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

        for ii,idx_vals in enumerate(indices):
            # Construct features for each problem in this batch
            # Placeholder inputs for computing class scores for all problems in a batch
            ff_inputs = torch.zeros((BATCH_SIZE*self.problem.n_obs, self.n_features)).to(device=self.device)
            cnn_inputs = torch.zeros((BATCH_SIZE*self.problem.n_obs, 3, self.problem.H, self.problem.W)).to(device=self.device)

            for prb_idx, idx_val in enumerate(idx_vals):
                prob_params = {}
                for k in params:
                    prob_params[k] = params[k][idx_val]

                # Feedforward inputs for each problem are identical and repeated self.problem.n_obs times
                ff_inp = torch.from_numpy(self.problem.construct_features(prob_params, self.prob_features))
                ff_inputs[self.problem.n_obs*prb_idx:self.problem.n_obs*(prb_idx+1)] = ff_inp.repeat(self.problem.n_obs,1).float().to(device=self.device)

                # CNN
                for ii_obs in range(self.problem.n_obs):
                    cnn_inputs[self.problem.n_obs*prb_idx+ii_obs] = torch.from_numpy(self.problem.construct_cnn_features(prob_params, self.prob_features, \
                                    ii_obs=ii_obs)).float().to(device=self.device)

            # Use strategy classifier to identify high ranking strategies for each feature
            class_scores = model(cnn_inputs, ff_inputs).detach().cpu().numpy()
            class_labels = np.argsort(class_scores, axis=1)[:,-self.n_evals:][:,::-1] # Predicted strategy index for each features; size BATCH_SIZE*self.problem.n_obs x self.n_evals

            # Loop through strategy dictionary once and save binary solution from strategies
            cl_idxs = np.unique(class_labels.flatten())
            obs_strats = {}   # Dictionary where each (k,v) pair gives (index of a strategy, binary solution)
            for cl_idx in cl_idxs:
                cl_ii = np.where(self.labels[:,0] == cl_idx)[0][0]
                obs_strats[cl_idx] = self.labels[cl_ii, 1:]

            # Generate Cartesian product of strategy combinations
            vv = [np.arange(0,self.n_evals) for _ in range(self.problem.n_obs)]
            strategy_tuples = list(itertools.product(*vv))
            # Sample from candidate strategy tuples based on "better" combinations
            probs_str = [1./(np.sum(st)+1.) for st in strategy_tuples]  # lower sum(st) values --> better
            probs_str = probs_str / np.sum(probs_str)

            # Need to identify which rows of ff_inputs & cnn_inputs are feasible and which are infeasible
            feature_is_feasible = [False]*ff_inputs.shape[0]    # True if feas, False if infeasible
            feature_str_idxs = [0]*ff_inputs.shape[0]           # Which strategy was used for a feature

            for prb_idx, idx_val in enumerate(idx_vals):
                prob_params = {}
                for k in params.keys():
                    prob_params[k] = params[k][idx_val]

                str_idxs = np.random.choice(np.arange(0,len(strategy_tuples)), max_evals, p=probs_str)
                # Manually add top-scoring strategy tuples
                if 0 in str_idxs:
                    str_idxs = np.unique(np.insert(str_idxs, 0, 0))
                else:
                    str_idxs = np.insert(str_idxs, 0, 0)[:-1]

                strategy_tuples_prb = [strategy_tuples[ii] for ii in str_idxs]
                prob_success = False

                # grab class scores for this prb_idx; has self.problem.n_obs rows and self.n_evals columns
                ind_max = class_labels[self.problem.n_obs*prb_idx:self.problem.n_obs*(prb_idx+1)]

                for ii, str_tuple in enumerate(strategy_tuples_prb):
                    if prob_success:
                        continue

                    # Assemble guess for this strategy tuple
                    y_guess = -np.ones((4*self.problem.n_obs, self.problem.N-1))
                    for ii_obs in range(self.problem.n_obs):
                        # rows of ind_max correspond to ii_obs, column to desired strategy
                        y_obs = obs_strats[ind_max[ii_obs, str_tuple[ii_obs]]]
                        y_guess[4*ii_obs:4*(ii_obs+1)] = np.reshape(y_obs, (4,self.problem.N-1))
                    if (y_guess < 0).any():
                        print("Strategy was not correctly found!")
                        return False

                    try:
                        prob_success = self.problem.solve_pinned(prob_params, y_guess, solver=cp.MOSEK)[0]
                    except:
                        prob_success = self.problem.solve_pinned(prob_params, y_guess, solver=cp.GUROBI)[0]

                    if prob_success:
                        # If feasible solution found, mark all of these features as feasible training points
                        for ii_obs in range(self.problem.n_obs):
                            feature_is_feasible[self.problem.n_obs*prb_idx+ii_obs] = True
                            feature_str_idxs[self.problem.n_obs*prb_idx+ii_obs] = str_tuple[ii_obs]
                    else:
                        # If infeasible solution found, mark all of these features as infeasible training points
                        for ii_obs in range(self.problem.n_obs):
                            feature_is_feasible[self.problem.n_obs*prb_idx+ii_obs] = False
                            feature_str_idxs[self.problem.n_obs*prb_idx+ii_obs] = str_tuple[ii_obs]

            scores = feas_model(cnn_inputs, ff_inputs)
            losses, labels = torch.zeros(scores.shape[0]).to(device=self.device), torch.zeros(scores.shape[0]).to(device=self.device)

            feas_idxs, infeas_idxs = np.where(feature_is_feasible)[0], np.where(np.logical_not(feature_is_feasible))[0]

            # If solution was feasible, label = 1., if infeasible label = -1.
            labels[feas_idxs] = 1.
            labels[infeas_idxs] = -1.

            # feas_loss = torch.zeros(1).to(device=self.device)
            for feas_idx in feas_idxs:
                # feas_loss += scores[feas_idx, feature_str_idxs[feas_idx]]
                losses[feas_idx] += scores[feas_idx, feature_str_idxs[feas_idx]]

            # infeas_loss = torch.zeros(1).to(device=self.device)
            for infeas_idx in infeas_idxs:
                # infeas_loss += scores[infeas_idx, feature_str_idxs[infeas_idx]]
                losses[infeas_idx] = scores[feas_idx, feature_str_idxs[feas_idx]]

            # TODO(acauligi): decide whether to penalize infeasible solutions or not
            inner_loss = inner_training_loss(losses, labels).to(device=self.device)

            # Descent step on feas_model network weights
            grad = torch.autograd.grad(inner_loss, feas_model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, feas_model.parameters())))

            # self.copy_all_but_last(model, feas_model.vars)

        return True
