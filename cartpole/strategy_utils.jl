function which_M(x,uc,sc,Y,Ak,Bk,Q,R,N,uc_min,uc_max,sc_min,sc_max,x_min,x_max,delta_min,delta_max,ddelta_min,ddelta_max, params, X0, Xg; eq_tol=1e-5,ineq_tol=1e-5)
  """ Implementation of cart-pole system with contacts "From Warm Start of Mixed-Integer Programs for Model Predictive Control of Hybrid Systems, Marcucci & Tedrake (2019)" """
  # params
  n,m = params["n"], params["m"]
  kappa = params["kappa"]
  nu = params["nu"]
  l = params["l"]
  dist = params["dist"]

  violations = Int64[]

  # Binary variable constraints
  for kk in 1:N-1
    for jj in 1:2
      # Check for when Eq. (27) is strict equality
      if jj == 1
        d_k    = -x[1,kk] + l*x[2,kk] - dist
        dd_k   = -x[3,kk] + l*x[4,kk]
      else
        d_k    =  x[1,kk] - l*x[2,kk] - dist
        dd_k   =  x[3,kk] - l*x[4,kk]
      end

      if abs(sc[jj,kk]-kappa*d_k-nu*dd_k) <= eq_tol
        push!(violations, 4*(kk-1) + 2*jj-1)
        push!(violations, 4*(kk-1) + 2*jj)
      end
    end
  end
  violations
end
function which_M(prob_dicts,dir_idx,prob_idx)
  data = prob_dicts[dir_idx]
  which_M(
    data["X"][prob_idx], data["Uc"][prob_idx],
    data["Sc"][prob_idx], data["Y"][prob_idx],
    data["prob"]["Ak"],data["prob"]["Bk"],data["prob"]["Q"],data["prob"]["R"],data["prob"]["N"],
    data["prob"]["uc_min"],data["prob"]["uc_max"],data["prob"]["sc_min"],data["prob"]["sc_max"],
    data["prob"]["x_min"],data["prob"]["x_max"],data["prob"]["delta_min"],data["prob"]["delta_max"],
    data["prob"]["ddelta_min"],data["prob"]["ddelta_max"],data["prob"]["params"],
    data["X0"][prob_idx], data["Xg"][prob_idx])
end
