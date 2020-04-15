import cvxpy as cp
import numpy as np
import mosek

n,m = 4,3
N = 11 
dh = 0.1
n_obs = 8

x0 = np.array([-0.39844222352768077, 0.20543510691704736, 0.24126227864050964, 0.4491739326585802])
xg = np.zeros(n)

Q = np.diag(np.array([10.,10,10,10]))
R = np.diag(np.array([1.,0,0]))

l = 1.
mc = 1.
mp = 1.
g = 10.
dh = 0.05
kappa,nu = 100., 30.

# state bounds
dist = 0.5
x_max = np.array([dist, np.pi/8, 2, 1])
x_min = -x_max

delta_min = np.array([
    -x_max[0] + l*x_min[1] - dist,
    x_min[0] - l*x_max[1] - dist
])
ddelta_min = np.array([
    -x_max[2] + l*x_min[3],
    x_min[2] - l*x_max[3]
])

delta_max = np.array([
    -x_min[0] + l*x_max[1] - dist,
    x_max[0] - l*x_min[1] - dist
])
ddelta_max = np.array([
    -x_min[2] + l*x_max[3],
    x_max[2] - l*x_min[3]
])

# control bounds
uc_min = -2.
uc_max = -uc_min
sc_min = kappa*delta_min + nu*ddelta_min
sc_max = kappa*delta_max + nu*ddelta_max

# dynamics
A = np.vstack((
    np.hstack((np.zeros((2,2)), np.eye(2))),
    np.zeros((2,4))
))
A[2,1] = g*mp/mc
A[3,1] = g*(mc+mp)/(l*mc)
B = np.zeros((n,m))
B[2,0] = 1/mc
B[3,:] = np.array([1/(l*mc), -1/(l*mp), 1/(l*mp)])
Ak = np.eye(n) + dh*A
Bk = dh*B

# Decision variables
cons = []

x = cp.Variable((n,N))
u = cp.Variable((m, N-1))
y = cp.Variable((4, N-1), integer=True)

# Initial condition
cons +=  [x[:,0] == x0]

# Dynamics constraints
for kk in range(N-1):
  cons += [x[:,kk+1] - (Ak @ x[:,kk] + Bk @ u[:,kk]) == np.zeros(n)]

# State and control constraints
for kk in range(N):
  cons += [x_min - x[:,kk] <= np.zeros(n)] 
  cons += [x[:,kk] - x_max <= np.zeros(n)]

for kk in range(N-1):
  cons += [uc_min - u[0,kk] <= 0.]
  cons += [u[0,kk] - uc_max <= 0]

# Binary variable constraints
for kk in range(N-1):
  for jj in range(2):
    if jj == 0:
      d_k    = -x[0,kk] + l*x[1,kk] - dist
      dd_k   = -x[2,kk] + l*x[3,kk]
    else:
      d_k    =  x[0,kk] - l*x[1,kk] - dist
      dd_k   =  x[2,kk] - l*x[3,kk]

    y_l = y[2*jj,kk]
    y_r = y[2*jj+1,kk]    
    d_min,d_max = delta_min[jj],delta_max[jj]
    dd_min,dd_max = ddelta_min[jj],ddelta_max[jj]
    f_min,f_max = sc_min[jj],sc_max[jj]

    # Eq. (26a)
    cons += [d_min*(1-y_l) <= d_k]
    cons += [d_k <= d_max*y_l]

    # Eq. (26b)
    cons += [f_min*(1-y_r) <= kappa*d_k + nu*dd_k]
    cons += [kappa*d_k + nu*dd_k <= f_max*y_r]

    # Eq. (27)
    cons += [nu*dd_max*(y_l-1) <= u[jj+1,kk] - kappa*d_k-nu*dd_k]
    cons += [u[jj+1,kk] - kappa*d_k-nu*dd_k <= f_min*(y_r-1)]

    cons += [-u[jj+1,kk] <= 0]
    cons += [u[jj+1,kk] <= f_max*y_l]
    cons += [u[jj+1,kk] <= f_max*y_r]

# LQR cost
lqr_cost = 0.
for kk in range(N):
  lqr_cost += cp.quad_form(x[:,kk]-xg, Q)
for kk in range(N-1):
  lqr_cost += cp.quad_form(u[:,kk],R)

prob = cp.Problem(cp.Minimize(lqr_cost), cons)
prob.solve()

# return prob, x, u, y 
