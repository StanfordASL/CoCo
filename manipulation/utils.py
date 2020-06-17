import cvxpy as cp
import numpy as np
from halton_sampling import generate_halton_samples

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
