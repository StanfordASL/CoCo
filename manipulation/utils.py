import numpy as np
from .halton_sampling import generate_halton_samples

def skew(v):
  out = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
  return out

def align_z(v):
  # function which takes a vector v and returns rotation
  # which aligns z-axis w/ v
  z = np.array([0.,0,1])
  u = skew(z) @ v
  s = np.linalg.norm(u, ord=2)
  c = v.T.dot(z)

  if s == 0:
    if c > 0.:
      return np.eye(3)
    else:
      return np.diag([-1,-1,1])
  
  R = np.eye(3) + skew(u) + skew(u) @ skew(u) * (1-c)/s**2
  return R

def cylinder_grasp_from_normal(v,h,r):
  # takes in an outward normal vector v (from origin), height h, and radius r;
  # returns portion of grasp matrix Gi corresponding to this normal

  x, y, z = v
  n_z = np.array([0,0,1.])

  # check intersection w/ tops
  if z > 0.:
    # solve for ray length
    t = h/(2*z)
    if np.linalg.norm(np.array([t*x, t*y])) <= r:
      p = t*v
      R = align_z(-n_z)
      return np.vstack((R, skew(p) @ R)), R, p
  elif z < 0:
    t = -h/(2*z)
    if np.linalg.norm(np.array([t*x, t*y])) <= r:
      p = t*v
      R = align_z(n_z)
      return np.vstack((R, skew(p) @ R)), R, p

  # if no intersection, then solve for the side
  t = r/np.linalg.norm(np.array([x,y]))
  p = t*v
  norm_in = -np.array([x,y,0.]) / np.linalg.norm(np.array([x,y,0.]))
  R = align_z(norm_in)
  return np.vstack((R, skew(p) @ R)), R, p

def sample_points(N_v, N_h, h=2, r=1, e_noise=0.05, rng=92):
  N = N_h * N_v
  eps = 0.025
  u = 2 * np.linspace(eps, 1-eps, N_v) - 1.
  th = 2*np.pi * np.linspace(0, (N_h-1.)/float(N_h), N_h)

  TH = np.kron(th, np.ones(N_v))
  U = np.kron(np.ones(int(N_h/2)), np.hstack((u, u[::-1])))

  randfloat = generate_halton_samples(3, N)

  x = np.sqrt(1 - U**2)*np.cos(TH) + e_noise*randfloat[0,:]
  y = np.sqrt(1 - U**2)*np.sin(TH) + e_noise*randfloat[1,:]
  z = U + e_noise*randfloat[2,:]

  lengths = np.linalg.norm(np.vstack((x,y,z)), axis=0)
  x /= lengths
  y /= lengths
  z /= lengths

  G, p = [], []
  for ii in range(N):
    Gr, R, p_i = cylinder_grasp_from_normal(np.array([x[ii],y[ii],z[ii]]),h,r)
    G.append(Gr)
    p.append(p_i)
  return G, p

def manipulation_prob(N_v, N_h, num_grasps, w, h, r, mu):
  # Implementation of manipulation system with contacts From "Fast Computation of Optimal Contact Forces, Boyd & Wegbreit (2007)"

  N = N_v * N_h # total number of points

  # sample points on cylinder
  Gr, p = sample_points(N_v, N_h, h, r)

  # Grasp optimization w.r.t. task specification
  V = np.hstack((np.eye(6), -np.eye(6)))

  # max normal force
  F_max = 1.

  # choose points
  G, p = sample_points(N_v, N_h, h, r, 0.02)

  # setup problem
  Y = cp.Variable(N, boolean=True) # indicator of if point is used
  a = cp.Variable(12) # magnitudes of achievable wrenches in each basis dir.

  # contact forces; one for each point, for each basis direction
  f = [cp.Variable((3,12)) for _ in range(N)]

  # maximize weighted polyhedron volume
  obj = w.T @ a

  cons = []

  # ray coefficients must be positive
  cons += [a >= 0.]

  for ii in range(12):
    # generate wrench in basis dir.
    cons += [cp.sum([G[jj]@f[jj][:,ii] for jj in range(N)]) == a[ii]*V[:,ii]]

    # friction cone
    for jj in range(N):
      cons += [cp.norm(f[jj][:2,ii]) <= mu*f[jj][2,ii]]

    for jj in range(N):
      cons += [f[jj][2,ii] <= F_max*Y[jj]]

  # limit number of chosen grasps
  cons += [cp.sum(Y) <= num_grasps]

  prob = cp.Problem(cp.Maximize(obj), cons)
  return prob
