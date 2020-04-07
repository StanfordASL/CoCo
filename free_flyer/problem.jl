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

function c2d(Ac, Bc, Ts)
  n = size(Ac, 1)
  m = size(Bc, 2)
  sysd = exp(Ts*[Ac Bc; zeros(m,n + m)])
  A = sysd[1:n,1:n]
  B = sysd[1:n,n + 1:n + m]
  return A, B
end

function obstacleAvoidanceProb(A, B, Q, R, N, umin, umax, velmin, velmax, posmin, posmax, obstacles, X0, Xg; verbose::Bool=false, optimizer=Gurobi.Optimizer)
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

  n::Int64 = size(A,1)/2
  m::Int64 = size(B,2)

  strategy_cons = []

  # Variables
  @variable(prob, x[1:2*n, 1:N]) # state
  @variable(prob, u[1:m, 1:N-1]) # control
  @variable(prob, t_u[1:1,1:N-1])  # control constraint

  # Logic variables
  nobstacles = size(obstacles)[1]
  @variable(prob, y[1:4*nobstacles, 1:N-1], Bin) # binary variables, with no initial constraints on integrality

  # Dynamics constraints
  for i in 1:N-1
    @constraint(prob, x[:,i+1] .== A*x[:,i] + B*u[:,i])
  end

  # Avoid constraints
  M = 100. # big M value
  for (i_obs, obstacle) in enumerate(obstacles)
    for i_dim in 1:n
      for i_t in 1:N-1
        o_min = obstacle[i_dim][1]
        yvar_min = 4*(i_obs - 1) + 2*(i_dim - 1) + 1
        o_max = obstacle[i_dim][2]
        yvar_max = 4*(i_obs - 1) + 2*i_dim
        @constraint(prob, x[i_dim,i_t+1] <= o_min + M*y[yvar_min,i_t])
        @constraint(prob, -x[i_dim,i_t+1] <= -o_max + M*y[yvar_max,i_t])
      end
    end

    for i_t in 1:N-1
      yvar_min = 4*(i_obs - 1) + 1
      yvar_max = 4*i_obs
      @constraint(prob, sum([y[ii, i_t] for ii in yvar_min:yvar_max]) <= 3)
    end
  end

  # Initial condition
  @constraint(prob, x[:,1] .== X0)

  # Region bounds
  for kk in 1:N
    for jj in 1:n
      con = @constraint(prob, posmin[jj] - x[jj,kk] <= 0.)
      push!(strategy_cons, con)
      con = @constraint(prob, x[jj,kk] - posmax[jj] <= 0.)
      push!(strategy_cons, con)
    end
  end

  # Velocity constraints
  for kk in 1:N
    for jj in 1:n
      con = @constraint(prob, velmin - x[n+jj, kk] .<= 0.)
      push!(strategy_cons, con)
      con = @constraint(prob, x[n+jj,kk] - velmax .<= 0.)
      push!(strategy_cons, con)
    end
  end

  unorm_lim = min(abs(umin),abs(umax))
  @constraint(prob, t_u .<= unorm_lim)
  # Control constraints
  for kk in 1:N-1
    con = @constraint(prob, [t_u[kk]; u[:,kk]] in SecondOrderCone())
    push!(strategy_cons, con)
  end

  # l2-norm of cost
  lqr_cost = 0.
  for kk = 1:N
    lqr_cost += (x[:,kk]-Xg)' * Q * (x[:,kk]-Xg)
  end
  for kk = 1:N-1
    lqr_cost += u[:,kk]' * R * u[:,kk]
  end

  @objective(prob, Min, lqr_cost)

  return prob, x, u, y, strategy_cons
end


function getObstacleAvoidanceProb(data, ii; verbose::Bool=false, optimizer=Gurobi.Optimizer)
  """ Given the dict containing all data, for a given data sample i return the 
  corresponding JuMP problem associated with it. """

  # Create the JuMP problem
  obstacleAvoidanceProb(data["prob"]["A"], data["prob"]["B"],
    data["prob"]["Q"], data["prob"]["R"], data["prob"]["N"],
    data["prob"]["umin"], data["prob"]["umax"],
    data["prob"]["velmin"], data["prob"]["velmax"],
    data["prob"]["posmin"], data["prob"]["posmax"],
    data["O"][ii], data["X0"][ii], data["Xg"][ii];
    verbose=verbose, optimizer=optimizer)
end

    
function solveObstacleAvoidanceProb(A, B, Q, R, N, umin, umax, velmin, velmax, posmin, posmax, obstacles, X0, Xg; optimizer=Gurobi.Optimizer)
  """ Solves the problem given by prob """

  # Create the JuMP problem
  prob, x, u, y, strategy_cons = obstacleAvoidanceProb(A, B, Q, R, N, umin, umax, velmin, velmax, posmin, posmax, obstacles, X0, Xg; optimizer=optimizer)

  original_stdout = stdout
  (read_pipe, write_pipe) = redirect_stdout()
  optimize!(prob)
  redirect_stdout(original_stdout)
  close(write_pipe)

  # Check and output solution
  if termination_status(prob) == MOI.OPTIMAL
    return value.(x), value.(u), value.(y), value(objective_function(prob)), MOI.get(prob, MOI.SolveTime()), MOI.get(prob, MOI.NodeCount()), true
  else
    return [], [], [], Inf, Inf, Inf, false
  end
end


function upperBoundObstacleAvoidanceProb(; verbose::Bool=false, optimizer=Gurobi.Optimizer)
  """ Computes an upper bound on the problem, without regarding X0 or obstacles """
  return 2000., true
end
