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
                    y_guesses = np.zeros((NUM_META_PROBLEMS, 4*self.problem.n_obs, self.problem.N-1))

                    # Compute feature vectors for each of the indices in idx_vals
                    # Note each problem itself has self.problem.n_obs task features
                    ff_inputs_inner = torch.zeros((NUM_META_PROBLEMS*self.problem.n_obs, self.n_features)).to(device=self.device)
                    cnn_inputs_inner = torch.zeros((NUM_META_PROBLEMS*self.problem.n_obs, 3, self.problem.H, self.problem.W)).to(device=self.device)

                    for ii_meta, idx_val in enumerate(idx_vals):
                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][idx_val]
                        prob_params_list.append(prob_params)

                        ff_inner = torch.from_numpy(self.problem.construct_features(prob_params, self.prob_features))
                        ff_inputs_inner[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)] = Variable(ff_inner.repeat(self.problem.n_obs,1)).float().to(device=self.device)

                        X_cnn_inner = np.zeros((self.problem.n_obs, 3, self.problem.H, self.problem.W))
                        for ii_obs in range(self.problem.n_obs):
                            X_cnn_inner[ii_obs] = self.problem.construct_cnn_features(prob_params, \
                                            self.prob_features, \
                                            ii_obs=ii_obs)
                        cnn_inputs_inner[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)] = Variable(torch.from_numpy(X_cnn_inner)).float().to(device=self.device)

                    class_scores = model(cnn_inputs_inner, ff_inputs_inner).detach().cpu().numpy()
                    class_labels = np.argmax(class_scores, axis=1)      # Predicted strategy index for each features of length NUM_META_PROBLEMS*self.problem.n_obs

                    feas_scores = feas_model(cnn_inputs_inner, ff_inputs_inner, vars=fast_weights)

                    for ii_meta in range(NUM_META_PROBLEMS):
                        # Grab strategy indices for a particular problem
                        class_labels_prb = class_labels[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)]
                        for cl_ii, cl in enumerate(class_labels_prb):
                            cl_idx = np.where(self.labels[:,0] == cl)[0][0]
                            y_obs = self.labels[cl_idx, 1:]
                            y_guesses[ii_meta, 4*cl_ii:4*(cl_ii+1)] = np.reshape(y_obs, (4, self.problem.N-1))

                    prob_success_dict = {}
                    for ii_meta in range(NUM_META_PROBLEMS):
                        prob_success_dict[ii_meta] = False
                        try:
                            prob_success_dict[ii_meta] = self.problem.solve_pinned(prob_params_list[ii_meta], y_guesses[ii_meta], solver=cp.MOSEK)[0]
                        except:
                            prob_success_dict[ii_meta] = self.problem.solve_pinned(prob_params_list[ii_meta], y_guesses[ii_meta], solver=cp.GUROBI)[0]

                    # self.model.share_memory()
                    # manager = mp.Manager()
                    # prob_success_dict = manager.dict()
                    # processes = []
                    # for process_idx in range(NUM_META_PROBLEMS):
                    #     p = mp.Process(target=solve_pinned_worker, \
                    #       args=[prob_params_list, y_guesses, process_idx, prob_success_dict])
                    #     p.start()
                    #     processes.append(p)
                    # for p in processes:
                    #     p.join()

                    for ii_meta in range(NUM_META_PROBLEMS):
                        # Grab strategy indices and feasibility scores for a particular problem
                        class_labels_prb = class_labels[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)]
                        feas_scores_prb = feas_scores[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)]

                        feas_loss = torch.zeros(1).to(device=self.device)
                        for ii_obs in range(self.problem.n_obs):
                            feas_loss += feas_scores_prb[ii_obs, class_labels_prb[ii_obs]]

                        if prob_success_dict[ii_meta]:
                            # If problem feasible, push scores to positive value
                            inner_loss += torch.relu(self.margin - feas_loss)
                        else:
                            # If problem infeasible, push scores to negative value
                            inner_loss += torch.relu(self.margin + feas_loss)
                    inner_loss /= float(NUM_META_PROBLEMS)

                    # Descent step on feas_model network weights
                    grad = torch.autograd.grad(inner_loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                # # Pass inner loop weights to MLOPT classifier (except last layer)
                # self.copy_all_but_last(model, fast_weights)

                ff_inputs = Variable(torch.from_numpy(X[idx,:])).float().to(device=self.device)
                labels = Variable(torch.from_numpy(Y[idx])).long().to(device=self.device)

                # forward + backward + optimize
                X_cnn = np.zeros((len(idx), 3,self.problem.H,self.problem.W))
                for idx_ii, idx_val in enumerate(idx):
                    prob_params = {}
                    for k in params:
                        prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                    X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                cnn_inputs = Variable(torch.from_numpy(X_cnn)).float().to(device=self.device)

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
                # fast_weights = list(feas_model.parameters())
                # model_weights = list(model.parameters())
                # for param_ii in range(len(fast_weights)-2):
                #     fast_weights[param_ii].data.copy_(model_weights[param_ii].detach())
                # fast_weights[-2].data.copy_(fast_weights[-2].detach())
                # fast_weights[-1].data.copy_(fast_weights[-1].detach())

                if itr % CHECKPOINT_AFTER == 0:
                    rand_idx = list(np.arange(0,X.shape[0]-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    ff_inputs = Variable(torch.from_numpy(X[test_inds,:])).float().to(device=self.device)
                    labels = Variable(torch.from_numpy(Y[test_inds])).long().to(device=self.device)

                    # forward + backward + optimize
                    X_cnn = np.zeros((len(test_inds), 3,self.problem.H,self.problem.W))
                    for idx_ii, idx_val in enumerate(test_inds):
                        prob_params = {}
                        for k in params:
                            prob_params[k] = params[k][self.cnn_features_idx[idx_val][0]]
                        X_cnn[idx_ii] = self.problem.construct_cnn_features(prob_params, self.prob_features, ii_obs=self.cnn_features_idx[idx_val][1])
                    cnn_inputs = Variable(torch.from_numpy(X_cnn)).float().to(device=self.device)
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

        BATCH_SIZE = self.training_params['BATCH_SIZE']
        rand_idx = list(np.arange(0,n_test))
        indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

        for ii,idx_vals in enumerate(indices):
            # Construct features for each problem in this batch
            # Placeholder inputs for computing class scores for all problems in a batch
            ff_inputs = torch.zeros((BATCH_SIZE*self.problem.n_obs, self.n_features)).to(device=self.device)
            cnn_inputs = torch.zeros((BATCH_SIZE*self.problem.n_obs, 3, self.problem.H, self.problem.W)).to(device=self.device)

            for ii_meta, idx_val in enumerate(idx_vals):
                prob_params = {}
                for k in params:
                    prob_params[k] = params[k][idx_val]

                # Feedforward inputs for each problem are identical and repeated self.problem.n_obs times
                ff_inp = torch.from_numpy(self.problem.construct_features(prob_params, self.prob_features))
                ff_inputs[self.problem.n_obs*ii_meta:self.problem.n_obs*(ii_meta+1)] = Variable(ff_inp.repeat(self.problem.n_obs,1)).float().to(device=self.device)

                # CNN
                for ii_obs in range(self.problem.n_obs):
                    cnn_inputs[self.problem.n_obs*ii_meta+ii_obs] = torch.from_numpy(self.problem.construct_cnn_features(prob_params, self.prob_features, \
                                    ii_obs=ii_obs)).float().to(device=self.device)

            class_scores = model(cnn_inputs, ff_inputs).detach().cpu().numpy()
            class_labels = np.argsort(class_scores, axis=1)[:,-self.n_evals:][:,::-1] # Predicted strategy index for each features; size BATCH_SIZE*self.problem.n_obs x self.n_evals

            # Loop through strategy dictionary once and save strategies
            uniq_idxs = np.unique(class_labels.flatten())
            obs_strats = {}
            for cl_idx in uniq_idxs:
                cl_ii = np.where(self.labels[:,0] == cl_idx)[0][0]
                obs_strats[cl_idx] = self.labels[cl_ii, 1:]

            # Generate Cartesian product of strategy combinations
            vv = [np.arange(0,self.n_evals) for _ in range(self.problem.n_obs)]
            strategy_tuples = list(itertools.product(*vv))
            # Sample from candidate strategy tuples based on "better" combinations
            probs_str = [1./(np.sum(st)+1.) for st in strategy_tuples]  # lower sum(st) values --> better
            probs_str = probs_str / np.sum(probs_str)

            # Need to identify which rows of ff_inputs & cnn_inputs are feasible and which are infeasible
            feas_ct, infeas_ct = 0, 0

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

                # dictionary to track whether a feature/obstacle pair ever leads to a feasible soln
                feas_str_dict = {}

                # grab class scores for this prb_idx; has self.problem.n_obs rows and self.n_evals columns
                ind_max = class_labels[self.problem.n_obs*prb_idx:self.problem.n_obs*(prb_idx+1)]

                for ii, str_tuple in enumerate(strategy_tuples_prb):
                    if prob_success:
                        continue

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
                            feas_str_dict[(feas_ct+ii_obs, str_tuple[ii_obs])] = True
                        feas_ct += self.problem.n_obs
                    else:
                        # If infeasible solution found, mark all of these features as infeasible training points
                        for ii_obs in range(self.problem.n_obs):
                            feas_str_dict[(infeas_ct+ii_obs, str_tuple[ii_obs])] = False
                        infeas_ct += self.problem.n_obs

                # Work back to identify which indices of input features correspond to feasible and infeasible solutions
                feas_indices, infeas_indices = [], []
                for key, feas in feas_str_dict.items():
                    ct, str_idx = key
                    if feas:
                        feas_indices += [ct]
                    else:
                        infeas_indices += [ct]

                ff_feas_features, cnn_feas_features = ff_inputs[feas_indices], cnn_inputs[feas_indices]
                ff_infeas_features, cnn_infeas_features = ff_inputs[infeas_indices], cnn_inputs[infeas_indices]

                feas_scores = feas_model(cnn_feas_features, ff_feas_features)
                infeas_scores = feas_model(cnn_infeas_features, ff_infeas_features)

                feas_loss = torch.zeros(1).to(device=self.device)
                infeas_loss = torch.zeros(1).to(device=self.device)
                for key, feas in feas_str_dict.items():
                    feas_ct, str_idx = key
                    if feas:
                        feas_loss += feas_scores[np.where(feas_ct == np.array(feas_indices))[0],str_idx]
                    else:
                        infeas_loss += infeas_scores[np.where(feas_ct == np.array(infeas_indices))[0],str_idx]

                inner_loss = torch.relu(self.margin - feas_loss) + torch.relu(self.margin + infeas_loss)
                inner_loss /= float(BATCH_SIZE)

                # Descent step on feas_model network weights
                grad = torch.autograd.grad(inner_loss, feas_model.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, feas_model.parameters())))

        return True
