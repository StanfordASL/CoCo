import osqp
import numpy as np
from scipy import sparse

class OsqpSolver():
    def __init__(self, free_flyer_prob, verbose=False):
        self.Ak, self.Bk = sparse.csc_matrix(free_flyer_prob.Ak), sparse.csc_matrix(free_flyer_prob.Bk)
        self.N = free_flyer_prob.N
        self.M_ = 100.0
        self.n, self.m = free_flyer_prob.n, free_flyer_prob.m
        self.n_obs = free_flyer_prob.n_obs
        self.posmin, self.posmax = free_flyer_prob.posmin, free_flyer_prob.posmax
        self.velmin, self.velmax = free_flyer_prob.velmin, free_flyer_prob.velmax
        self.umin, self.umax = free_flyer_prob.umin, free_flyer_prob.umax
        self.Q, self.R = free_flyer_prob.Q, free_flyer_prob.R
        self.verbose = verbose
        self.eps_abs, self.eps_rel = 1e-5, 1e-5

    def setup_prob(self, p_dict, y_opt):
        x0 = p_dict['x0']
        xg = p_dict['xg']
        obstacles = p_dict['obstacles']

        # Dynamics constraint
        Au = sparse.kron(sparse.eye(self.N, k=-1), self.Ak) - sparse.kron(sparse.eye(self.N), sparse.eye(2*self.n))
        Bu = sparse.kron(sparse.csc_matrix(np.vstack((np.zeros((1, self.N-1)), np.eye(self.N-1)))), self.Bk)
        
        Aeq = sparse.hstack([Au, Bu])
        leq = np.concatenate((-x0, np.zeros(2*self.n*(self.N-1))))
        ueq = leq

        # Box constraint
        x_min = np.concatenate((self.posmin, self.velmin*np.ones(self.n)))
        x_max = np.concatenate((self.posmax, self.velmax*np.ones(self.n)))

        u_min = self.umin*np.ones(self.m)
        u_max = self.umax*np.ones(self.m)

        Aineq = sparse.kron(sparse.eye(self.N-1), sparse.eye(2*self.n+self.m))
        Aineq = sparse.eye(2*self.n*self.N + self.m*(self.N-1))
        lineq = np.concatenate([
            np.kron(np.ones(self.N), x_min),
            np.kron(np.ones(self.N-1), u_min)
        ])
        uineq = np.concatenate([
            np.kron(np.ones(self.N), x_max),
            np.kron(np.ones(self.N-1), u_max)
        ])

        # Obstacle avoidance constraints
        mask_ = sparse.csc_matrix(np.hstack((np.eye(self.n), np.zeros((self.n,self.n)))))
        mask_ = sparse.kron(sparse.eye(self.N), mask_)
        A_mask = mask_
        for ii in range(self.n_obs-1):
            A_mask = sparse.vstack((A_mask, mask_))
        B_mask = sparse.csc_matrix(np.zeros((A_mask.shape[0], self.m*(self.N-1))))
        A_mask = sparse.hstack([A_mask, B_mask])

        lmask, umask = np.array([]), np.array([])
        for ii in range(self.n_obs):
            lmask, umask = np.append(lmask, x0[:self.n]), np.append(umask, x0[:self.n])
            y_opt_ii = y_opt[2*self.n*ii:2*self.n*(ii+1),:]
            obstacle = obstacles[:,ii]
            for jj in range(self.N-1):
                umask = np.append(umask, obstacle[0] + self.M_*y_opt_ii[0,jj])
                lmask = np.append(lmask, obstacle[1] - self.M_*y_opt_ii[1,jj])
                umask = np.append(umask, obstacle[2] + self.M_*y_opt_ii[2,jj])
                lmask = np.append(lmask, obstacle[3] - self.M_*y_opt_ii[3,jj])

        # Construct cost matrices
        Q, R = sparse.csc_matrix(self.Q), sparse.csc_matrix(self.R)
        P = sparse.block_diag([
            sparse.kron(sparse.eye(self.N), Q),
            sparse.kron(sparse.eye(self.N-1), R)
        ])
        q = np.hstack((
            np.kron(np.ones(self.N), -self.Q @ xg),
            np.zeros(self.m*(self.N-1))
        ))

        # Construct QP constraints and bounds
        A = sparse.vstack([Aeq, Aineq, A_mask], format='csc')
        self.l = np.hstack([leq, lineq, lmask])
        self.u = np.hstack([ueq, uineq, umask])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, self.l, self.u, warm_start=True,
            verbose=self.verbose, eps_abs=self.eps_abs, eps_rel=self.eps_rel)

    def update_prob(self, p_dict, y_opt):
        x0 = p_dict['x0']
        xg = p_dict['xg']
        obstacles = p_dict['obstacles']

        leq = np.concatenate((-x0, np.zeros(2*self.n*(self.N-1))))
        ueq = leq

        x_min = np.concatenate((self.posmin, self.velmin*np.ones(self.n)))
        x_max = np.concatenate((self.posmax, self.velmax*np.ones(self.n)))

        u_min = self.umin*np.ones(self.m)
        u_max = self.umax*np.ones(self.m)

        lineq = np.concatenate([
            np.kron(np.ones(self.N), x_min),
            np.kron(np.ones(self.N-1), u_min)
        ])
        uineq = np.concatenate([
            np.kron(np.ones(self.N), x_max),
            np.kron(np.ones(self.N-1), u_max)
        ])

        lmask, umask = np.array([]), np.array([])
        for ii in range(self.n_obs):
            lmask, umask = np.append(lmask, x0[:self.n]), np.append(umask, x0[:self.n])
            y_opt_ii = y_opt[2*self.n*ii:2*self.n*(ii+1),:]
            obstacle = obstacles[:,ii]
            for jj in range(self.N-1):
                umask = np.append(umask, obstacle[0] + self.M_*y_opt_ii[0,jj])
                lmask = np.append(lmask, obstacle[1] - self.M_*y_opt_ii[1,jj])
                umask = np.append(umask, obstacle[2] + self.M_*y_opt_ii[2,jj])
                lmask = np.append(lmask, obstacle[3] - self.M_*y_opt_ii[3,jj])

        self.prob.update(l=self.l, u=self.u)

    def solve(self):
        res = self.prob.solve()

        prob_success_osqp = res.info.status == 'solved'
        optvals = None
        if prob_success_osqp:
            x_osqp = np.zeros((2*self.n, self.N))
            u_osqp = np.zeros((self.m, self.N-1))
            for ii in range(self.N):
                x_osqp[:,ii] = res.x[2*self.n*ii:2*self.n*(ii+1)]
            for ii in range(self.N-1):
                u_osqp[:,ii] = res.x[2*self.n*self.N+self.m*ii:2*self.n*self.N+self.m*(ii+1)]
            optvals = (x_osqp, u_osqp)
        return prob_success_osqp, optvals
