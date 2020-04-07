using Convex
using Mosek, MosekTools
using Suppressor
using LinearAlgebra
using MathOptInterface
const MOI = MathOptInterface

include("halton_sampling.jl")

##Helper functions for Cylinder grasping
function skew(v)
    return [0 -v[3] v[2]
            v[3] 0 -v[1]
            -v[2] v[1] 0]
end

function align_z(v)
    #function which takes a vector v & returns rotation matrix which aligns z-axis w/ v
    z = [0, 0, 1]
    u = skew(z)*v
    s = norm(u)
    c = v'*z
    
    if s == 0
        if c > 0
            return I
        else
            return Diagonal([-1,-1,1])
        end
    end
    
    R = I + skew(u) + skew(u)*skew(u)*(1-c)/(s^2)
end

function cylinder_grasp_from_normal(v,h,r)
    #takes in an *outward* normal vector, v (from origin), height h, and radius r; 
    #returns portion of grasp matrix Gi corresponding to this normal
    x,y,z = v
    n_z = [0,0,1]
    
    #check intersection w/tops
    if z > 0
        t = h/(2*z) #solve for ray length
        if norm([t*x,t*y]) <= r
            p = t*v
            R = align_z(-n_z)
            return vcat(R,skew(p)*R), R, p
        end
    elseif z < 0
        t = -h/(2*z)
        if norm([t*x,t*y]) <= r
            p = t*v
            R = align_z(n_z)
            return vcat(R,skew(p)*R), R, p
        end
    end
    #if no intersection, then solve for the side
    t = r/norm([x,y])
    p = t*v
    norm_in = -[x,y,0]/norm([x,y,0])
    R = align_z(norm_in)
    return vcat(R,skew(p)*R), R, p
end

function sample_points(N_v,N_h,h=2,r=1,e_noise=0.05,rng=nothing)
  N = N_h * N_v
  eps = 0.025
  u = 2*collect(range(eps,stop=1-eps,length=N_v)) .- 1
  th = 2*pi*collect(range(0,stop=(N_h-1)/(N_h),length=N_h))

  TH = vcat(fill.(th,N_v)...)
  U = repeat(vcat(u,reverse(u)),Int(N_h/2))

  halton_rand = generateHaltonSamples(3,N)
  x = sqrt.(1 .- U.^2).*cos.(TH) .+ e_noise*halton_rand[1,:]
  y = sqrt.(1 .- U.^2).*sin.(TH) .+ e_noise*halton_rand[2,:]
  z = U .+ e_noise*halton_rand[3,:]

  lengths = sqrt.(sum(x.^2 + y.^2 + z.^2, dims=2))
  x = x ./ lengths
  y = y ./ lengths
  z = z ./ lengths

  G = []
  p = []
  for ind in 1:N
    Gr, R, p_i = cylinder_grasp_from_normal([x[ind],y[ind],z[ind]],h,r)
    push!(G,Gr)
    push!(p,p_i)
  end
  G, p
end

function manipulationProb(N_v,N_h,num_grasps,w,h,r,mu; verbose::Bool=false)
  """ Implementation of manipulation system with contacts From "Fast Computation of Optimal Contact Forces, Boyd & Wegbreit (2007)" """
    
  solver = Mosek.Optimizer(QUIET=verbose,PRESOLVE_USE=0,NUM_THREADS=1)

  N = N_v*N_h
  Gr, p = sample_points(N_v,N_h,h,r) # sample points on cylinder

  strategy_cons = []
  #Grasp optimization w/ task specifications
  V = hcat(Matrix(I,6,6),Matrix(-I,6,6)) #matrix of basis functions
    
  N = N_v*N_h #total number of points
  F_max = 1 #max normal force

  #choose points
  G, p = sample_points(N_v,N_h,h,r,0.02)
  #setup problem
  Y = Variable(N,:Bin) #indicator of if point is used
  a = Variable(12) #magnitudes of achievable wrenches in each basis dir.
  f = [Variable(3,12) for i = 1:N] #contact forces; one for each point, for each basis dir.
  prob = maximize(w'*a) #objective -- maximize weighted polyhedron volume
  prob.constraints += a >= 0 #ray coeffs. must be postive
  for i = 1:12
    prob.constraints += sum([G[j]*f[j][:,i] for j in 1:N]) == a[i]*V[:,i] #generate wrench in basis dir.
    prob.constraints += Constraint[norm(f[j][1:2,i]) <= mu*f[j][3,i] for j in 1:N] #friction cone
    prob.constraints += Constraint[f[j][3,i] <= F_max*Y[j] for j in 1:N] #indicator if point used
  end
  prob.constraints += sum(Y) <= num_grasps; #limit number of chosen grasps

  return prob, a, f, Y, strategy_cons, solver
end

function getManipulationProb(data, ii; verbose::Bool=false) 
  """ Given the dict containing all data, for a given data sample ii return the
  corresponding Convex problem associated with it. """
  data["prob"]["params"]["num_grasps"] = 4
  manipulationProb(
    data["prob"]["params"]["N_v"],data["prob"]["params"]["N_h"],data["prob"]["params"]["num_grasps"],
    data["ws"][ii],data["h"][ii],data["r"][ii],data["mu"][ii]; verbose=verbose) 
end

function solveManipulationProb(N_v,N_h,num_grasps,w,h,r,mu; verbose::Bool=false) 
  """ Solves the problem given by prob """
  prob, a, f, Y, strategy_cons, solver = manipulationProb(N_v,N_h,num_grasps,w,h,r,mu; verbose=verbose)
  @suppress begin
    solve!(prob,solver)
  end

  if prob.status == MathOptInterface.OPTIMAL
    solve_time = MOI.get(prob.model.model, MOI.SolveTime())
    node_count = MOI.get(prob.model.model, MOI.NodeCount())
    f_vals = cat([f[i].value for i in 1:N_v*N_h]...,dims=3)
    return a.value, f_vals, Y.value, prob.optval, solve_time, node_count, true
  else
    return [], [], [], Inf, Inf, Inf, false
  end
end

function upperBoundManipulationProb(; verbose::Bool=false)
  return 20000., true
end
