function which_M(x, u, A, B, Q, R, N, umin, umax, velmin, velmax, posmin, posmax, obstacles, X0, Xg; eq_tol=1e-5,ineq_tol=1e-5)
  ##takes in problem data & continuous assignment, returns list of tight big-M constraints, one per obstacle
  n::Int64 = size(A,1)/2
  m::Int64 = size(B,2)

  # Variables
  nobstacles = size(obstacles)[1]

  violations = [] #list of obstacle big-M violations
  for (i_obs, obstacle) in enumerate(obstacles)
    curr_violations = [] #violations for current obstacle
    for i_t in 1:N-1
      for i_dim in 1:n
        #fetch obstacle limits
        o_min = obstacle[i_dim][1]
        o_max = obstacle[i_dim][2]

        if (x[i_dim,i_t+1] - o_min > ineq_tol)
          push!(curr_violations, 2*(i_dim) -1 + 2*n*(i_t-1))
        end

        if (-x[i_dim,i_t+1] + o_max > ineq_tol)
          push!(curr_violations, 2*(i_dim) + 2*n*(i_t-1))
        end
      end
    end
    push!(violations,curr_violations)
  end
  violations
end
function which_M(prob_dicts,dir_idx,prob_idx)
  which_M(
    prob_dicts[dir_idx]["X"][prob_idx], prob_dicts[dir_idx]["U"][prob_idx],
    prob_dicts[dir_idx]["prob"]["A"], prob_dicts[dir_idx]["prob"]["B"],
    prob_dicts[dir_idx]["prob"]["Q"], prob_dicts[dir_idx]["prob"]["R"], prob_dicts[dir_idx]["prob"]["N"],
    prob_dicts[dir_idx]["prob"]["umin"], prob_dicts[dir_idx]["prob"]["umax"],
    prob_dicts[dir_idx]["prob"]["velmin"], prob_dicts[dir_idx]["prob"]["velmax"],
    prob_dicts[dir_idx]["prob"]["posmin"], prob_dicts[dir_idx]["prob"]["posmax"],
    prob_dicts[dir_idx]["O"][prob_idx], prob_dicts[dir_idx]["X0"][prob_idx], prob_dicts[dir_idx]["Xg"][prob_idx]
  )
end

function check_feasible(x, u, Y, strategy_cons, A, B, Q, R, N, umin, umax, velmin, velmax, posmin, posmax, obstacles, X0, Xg; eq_tol=1e-5,ineq_tol=1e-5)
  n::Int64 = size(A,1)/2
  m::Int64 = size(B,2)

  # Variables
  nobstacles = size(obstacles)[1]

  y = reshape(Y, N-1, 4*nobstacles)'
  strategy_cons = []

  # Dynamics constraints
  for ii in 1:N-1
    if norm(A*x[:,ii]+B*u[:,ii] - x[:,ii+1], 1) >= eq_tol
      return false
    end
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

        if (x[i_dim,i_t+1] - (o_min + M*y[yvar_min,i_t]) > ineq_tol)
          return false
        elseif (-x[i_dim,i_t+1] - (-o_max + M*y[yvar_max,i_t]) > ineq_tol)
          return false
        end
      end
    end

    for i_t in 1:N-1
      yvar_min = 4*(i_obs - 1) + 1
      yvar_max = 4*i_obs
      if sum([y[ii, i_t] for ii in yvar_min:yvar_max]) > 3
        return false
      end
    end
  end

  # Initial condition
  if norm(x[:,1] - X0,1) > eq_tol
    return false
  end

  str_ct = 1

  # Region bounds
  for kk in 1:N
    for jj in 1:n
      if (posmin[jj] - x[jj,kk] > ineq_tol)
        return false
      elseif str_ct in strategy_cons && (posmin[jj] - x[jj,kk] < -ineq_tol)
        return false
      end
      str_ct+=1

      if (x[jj,kk] - posmax[jj] > ineq_tol)
        return false
      elseif str_ct in strategy_cons && (x[jj,kk] - posmax[jj] < -ineq_tol)
        return false
      end
      str_ct+=1
    end
  end

  # Velocity constraints
  for kk in 1:N
    for jj in 1:n
      if (velmin - x[n+jj, kk] > ineq_tol)
        return false
      elseif str_ct in strategy_cons && (velmin - x[n+jj, kk] < -ineq_tol)
        return false
      end
      str_ct+=1

      if (x[n+jj,kk] - velmax > ineq_tol)
        return false
      elseif str_ct in strategy_cons && (x[n+jj,kk] - velmax < -ineq_tol)
      end
      str_ct+=1
    end
  end

  # Control constraints
  for kk in 1:N-1
    for jj in 1:m
      if (umin - u[jj,kk] > ineq_tol)
        return false
      elseif str_ct in strategy_cons && (umin - u[jj,kk] < -ineq_tol)
      end
      str_ct+=1

      if (u[jj,kk] - umax > ineq_tol)
        return false
      elseif str_ct in strategy_cons && (u[jj,kk] - umax < -ineq_tol)
        return false
      end
      str_ct+=1
    end
  end
  return true
end

function check_feasible(prob_dicts,dir_idx,prob_idx,y_strategy,active_cons)
  check_feasible(
    prob_dicts[dir_idx]["X"][prob_idx], prob_dicts[dir_idx]["U"][prob_idx],
    y_strategy, active_cons,
    prob_dicts[dir_idx]["prob"]["A"], prob_dicts[dir_idx]["prob"]["B"],
    prob_dicts[dir_idx]["prob"]["Q"], prob_dicts[dir_idx]["prob"]["R"], prob_dicts[dir_idx]["prob"]["N"],
    prob_dicts[dir_idx]["prob"]["umin"], prob_dicts[dir_idx]["prob"]["umax"],
    prob_dicts[dir_idx]["prob"]["velmin"], prob_dicts[dir_idx]["prob"]["velmax"],
    prob_dicts[dir_idx]["prob"]["posmin"], prob_dicts[dir_idx]["prob"]["posmax"],
    prob_dicts[dir_idx]["O"][prob_idx], prob_dicts[dir_idx]["X0"][prob_idx], prob_dicts[dir_idx]["Xg"][prob_idx]
  )
end
