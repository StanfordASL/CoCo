import mosek
import cvxpy as cp
import numpy as np
import sys

N = 6
dh = 0.75
n_obs = 8 
n = 2
m = 2

# Double integrator matrices
Ak = np.matrix([[1,dh], [0,1]])
Ak = np.kron(Ak, np.eye(n))
Bk = np.matrix([[0.5*dh**2], [dh]])
Bk = np.kron(Bk, np.eye(n))

cons = []

# Variables
x = cp.Variable((2*n,N)) # state
u = cp.Variable((m,N-1))  # control

# Logic variables
y = cp.Variable((4*n_obs,N-1), integer=True) # binary variables, with no initial constraints on integrality

# Double integrator matrices
Ak = np.matrix([[1,dh], [0,1]])
Ak = np.kron(Ak, np.eye(n))
Bk = np.matrix([[0.5*dh**2], [dh]])
Bk = np.kron(Bk, np.eye(n))

Q = np.diag(np.array([2.,2,1,1]))
R = np.diag(np.array([0.1,0.1]))

posmin = np.zeros(2)
ft2m = 0.3048
posmax = ft2m*np.array([12.,9])

velmin = -0.2
velmax = 0.2

mass_ff = 0.5*(15.36+18.08)
thrust_max = 2*1.
umax = thrust_max / mass_ff


# Parameters
x0 = np.array([3.2327651282903074, 1.8639889326405996, -0.03603204951417299, 0.1869893802945004 ])
xg = np.hstack((0.9*posmax, np.zeros(n)))

obstacles = np.zeros((4,n_obs))
obstacles[:,0] = np.array([1.2424206311085149, 2.018378801026382, 1.9680147331717865, 2.3938458246877725])
obstacles[:,1] = np.array([2.8444978100500102, 3.4223733537122962, 0.8743882162614238, 1.3867471415089896])
obstacles[:,2] = np.array([2.234562183271801, 2.6751572994146655, 0.8328177555395004, 1.258507360435613])
obstacles[:,3] = np.array([1.2424206311085149, 1.6862713872463595, 1.6952039828455434, 2.0336794262861475])
obstacles[:,4] = np.array([1.8549963125408497, 2.591768377676192, 0.8328177555395004, 1.5084740719325855])
obstacles[:,5] = np.array([1.8678186722249257, 2.206731894204943, 1.673194026679264, 2.0160116107815096])
obstacles[:,6] = np.array([1.6058401667009463, 2.1375152896627454, 0.8328177555395004, 1.1888081940185842])
obstacles[:,7] = np.array([2.8149088329154806, 3.140716143252644, 1.0704990822004385, 1.5983451080141902])

Q = np.diag(np.array([2,2,1,1]))
R = np.diag(np.array([0.1,0.1]))

posmin = np.zeros(2)
ft2m = 0.3048
posmax = ft2m*np.array([12.,9])

velmin = -0.2
velmax = 0.2

mass_ff = 0.5*(15.36+18.08)
thrust_max = 2*1.
umax = thrust_max / mass_ff

cons += [x[:,0] == x0]

# Dynamics constraints
for ii in range(N-1):
  cons += [x[:,ii+1] == Ak @ x[:,ii] + Bk @ u[:,ii]]

M = 100. # big M value
for i_obs in range(n_obs):
  for i_dim in range(n):
    o_min = obstacles[n*i_dim,i_obs]
    o_max = obstacles[n*i_dim+1,i_obs]

    for i_t in range(N-1):
      yvar_min = 4*i_obs + n*i_dim
      yvar_max = 4*i_obs + n*i_dim + 1

      cons += [x[i_dim,i_t+1] <= o_min + M*y[yvar_min,i_t]]
      cons += [-x[i_dim,i_t+1] <= -o_max + M*y[yvar_max,i_t]]

  for i_t in range(N-1):
    yvar_min, yvar_max = 4*i_obs, 4*(i_obs+1)
    cons += [sum([y[ii,i_t] for ii in range(yvar_min,yvar_max)])]

# Region bounds
for kk in range(N):
  for jj in range(n):
    cons += [posmin[jj] - x[jj,kk] <= 0]
    cons += [x[jj,kk] - posmax[jj] <= 0]

# Velocity constraints
for kk in range(N):
  for jj in range(n):
    cons += [velmin - x[n+jj,kk] <= 0]
    cons += [x[n+jj,kk] - velmax <= 0]
    
# Control constraints
for kk in range(N-1):
    cons += [cp.norm(u[:,kk]) <= umax]

cost = 0.
# l2-norm of cost
for kk in range(N):
    cost += cp.quad_form(x[:,kk]-xg, Q)

for kk in range(N-1):
    cost += cp.quad_form(u[:,kk], R)
cost

prob = cp.Problem(cp.Minimize(cost), cons)
prob.solve(solver=cp.MOSEK)
