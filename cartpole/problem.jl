using JuMP
using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
using SCS
using GLPK
using MosekTools
using Mosek
using Gurobi
using LinearAlgebra

function cartpoleProb(Ak,Bk,Q,R,N,uc_min,uc_max,sc_min,sc_max,x_min,x_max,delta_min,delta_max,ddelta_min,ddelta_max, params, X0, Xg; verbose::Bool=false, optimizer=Gurobi.Optimizer)
  """ Implementation of cart-pole system with contacts From "Warm Start of Mixed-Integer Programs for Model Predictive Control of Hybrid Systems, Marcucci & Tedrake (2019)" """
  original_stdout = stdout
  (read_pipe, write_pipe) = redirect_stdout()
  prob = Model(with_optimizer(optimizer))
  if optimizer == Gurobi.Optimizer
    prob = verbose ? Model(with_optimizer(optimizer, OutputFlag=1, FeasibilityTol=1e-9, Presolve=0, Threads=1)) : Model(with_optimizer(optimizer, OutputFlag=0, FeasibilityTol=1e-9, Presolve=0, Threads=1))
  elseif optimizer == GLPK.Optimizer
    prob = Model(with_optimizer(optimizer))
  elseif optimizer == Mosek.Optimizer
    prob = verbose ? Model(with_optimizer(optimizer, QUIET=false, PRESOLVE_USE=0, NUM_THREADS=1)) : Model(with_optimizer(optimizer, QUIET=true, PRESOLVE_USE=0, NUM_THREADS=1))
  elseif optimizer == SCS.Optimizer
    prob = verbose ? Model(with_optimizer(optimizer, verbose=1)) : Model(with_optimizer(optimizer, verbose=0))
  end
  redirect_stdout(original_stdout)
  close(write_pipe)

  # params
  n,m = params["n"], params["m"]
  kappa = params["kappa"]
  nu = params["nu"]
  l = params["l"]
  dist = params["dist"]

  strategy_cons = []

  # Decision variables
  @variable(prob, x[1:n,1:N])
  @variable(prob, uc[1:1,1:N-1])   # continuous input
  @variable(prob, sc[1:2,1:N-1])   # symbolic forces (left,right)
  @variable(prob, y[1:4, 1:N-1], Bin)

  # Initial condition
  @constraint(prob, x[:,1] - X0 .== zeros(n))
  
  # Dynamics constraints
  for kk = 1:N-1
    uk = [uc[kk]; sc[:,kk]]
    @constraint(prob, Ak*x[:,kk]+Bk*uk -x[:,kk+1] .== zeros(n))
  end
  
  # State and control constraints
  for kk = 1:N
    for ii in 1:n
      con = @constraint(prob, x_min[ii] - x[ii,kk] .<= 0.)
      push!(strategy_cons, con)
      con = @constraint(prob, x[ii,kk] - x_max[ii] .<= 0.)
      push!(strategy_cons, con)
    end
  end
  
  for kk = 1:N-1
    con = @constraint(prob, uc_min - uc[kk] <= 0.)
    push!(strategy_cons, con)
    con = @constraint(prob, uc[kk] - uc_max <= 0.)
    push!(strategy_cons, con)
  end
  
  # Binary variable constraints
  for kk in 1:N-1
    for jj in 1:2
      if jj == 1
        d_k    = -x[1,kk] + l*x[2,kk] - dist
        dd_k   = -x[3,kk] + l*x[4,kk]
      else
        d_k    =  x[1,kk] - l*x[2,kk] - dist
        dd_k   =  x[3,kk] - l*x[4,kk]
      end

      y_l, y_r = y[2*jj-1:2*jj,kk]
      d_min,d_max = delta_min[jj],delta_max[jj]
      dd_min,dd_max = ddelta_min[jj],ddelta_max[jj]
      f_min,f_max = sc_min[jj],sc_max[jj]

      # Eq. (26a)
      con = @constraint(prob, d_min*(1-y_l) <= d_k)
      con = @constraint(prob, d_k <= d_max*y_l)

      # Eq. (26b)
      con = @constraint(prob, f_min*(1-y_r) <= kappa*d_k+nu*dd_k)
      con = @constraint(prob, kappa*d_k+nu*dd_k <= f_max*y_r)

      # Eq. (27)
      con = @constraint(prob, nu*dd_max*(y_l-1) <= sc[jj,kk]-kappa*d_k-nu*dd_k)
      con = @constraint(prob, sc[jj,kk]-kappa*d_k-nu*dd_k <= f_min*(y_r-1))

      con = @constraint(prob, -sc[jj,kk] <= 0)
      con = @constraint(prob, sc[jj,kk] <= f_max*y_l)
      con = @constraint(prob, sc[jj,kk] <= f_max*y_r)
    end
  end
  
  # LQR cost
  lqr_cost = 0.
  for kk = 1:N
    lqr_cost += (x[:,kk]-Xg)' * Q * (x[:,kk]-Xg)
  end
  for kk = 1:N-1
    lqr_cost += uc[kk]*R[1,1]*uc[kk]
  end

  @objective(prob, Min, lqr_cost)

  return prob, x, uc, sc, y, strategy_cons
end

function getCartpoleProb(data, ii; verbose::Bool=false, optimizer=Gurobi.Optimizer)
  """ Given the dict containing all data, for a given data sample ii return the
  corresponding JuMP problem associated with it. """

  # Create the JuMP problem
  cartpoleProb(data["prob"]["Ak"],data["prob"]["Bk"],data["prob"]["Q"],data["prob"]["R"],data["prob"]["N"],
    data["prob"]["uc_min"],data["prob"]["uc_max"],data["prob"]["sc_min"],data["prob"]["sc_max"],
    data["prob"]["x_min"],data["prob"]["x_max"],data["prob"]["delta_min"],data["prob"]["delta_max"],
    data["prob"]["ddelta_min"],data["prob"]["ddelta_max"],data["prob"]["params"],
    data["X0"][ii], data["Xg"][ii];
    verbose=verbose, optimizer=optimizer)
end

function solveCartpoleProb(Ak,Bk,Q,R,N,uc_min,uc_max,sc_min,sc_max,x_min,x_max,delta_min,delta_max,ddelta_min,ddelta_max,params,X0,Xg; optimizer=Gurobi.Optimizer)
  """ Solves the problem given by prob """

  # Create the JuMP problem
  prob, x, uc, sc, y, strategy_cons = cartpoleProb(Ak,Bk,Q,R,N,uc_min,uc_max,sc_min,sc_max,x_min,x_max,delta_min,delta_max,ddelta_min,ddelta_max, params,X0, Xg; optimizer=optimizer)

  original_stdout = stdout
  (read_pipe, write_pipe) = redirect_stdout()
  optimize!(prob)
  redirect_stdout(original_stdout)
  close(write_pipe)
  
  # Check and output solution
  if termination_status(prob) == MOI.OPTIMAL
    return value.(x), value.(uc), value.(sc), value.(y), value(objective_function(prob)), MOI.get(prob, MOI.SolveTime()), MOI.get(prob, MOI.NodeCount()), true
  else
    return [], [], [], [], Inf, Inf, Inf, false
  end
end

function upperBoundCartpoleProb(; verbose::Bool=false, optimizer=Gurobi.Optimizer)
  return 500., true
end
