using LinearAlgebra

function vectorizeIntegerData(yArray)
  """ Takes the data from dictionary dict[-]["Y"][-] which is a
  (4,N) array and convert it into the correct vector form (column wise
  stacking) and casts as Ints. """
  yVector = yArray'[:]
  return Int.(round.(yVector))
end

function construct_bnb_features(nodes, bnb_prob, bnb_features)
  n_nodes = length(nodes)
  y_dim = length((nodes[1]).lbs)
  
  # Ensure that "node_assignments" at front if used
  if "node_assignments" in bnb_features
    bnb_features = reverse(push!(filter!(x->x≠"node_assignments", bnb_features), "node_assignments"))
  end

  feature_matrix = []
  for (idx, node) in enumerate(nodes)
    # Build feature vector for each node
    feature_vec = []
    for feature in bnb_features
      if feature == "node_LB"
        # Feature for lower bound for the given node
        phi_f = node.LB
      elseif feature == "node_assignments"
        # Looks at all integer variables assignments, and assigns -1, 0, or 1
        phi_f = zeros(y_dim)
        for ii in 1:y_dim
          if node.lbs[ii] == node.ubs[ii]
            phi_f[ii] = (node.lbs[ii] == 0) ? -1. : 1.
          end
        end
      elseif feature == "depth"
        phi_f = node.depth
      else
        # If feature not good
        println(string("Feature ",feature," is unknown."))
        continue
      end

      # Concatenate to the current vector
      feature_vec = isempty(feature_vec) ? phi_f : vcat(feature_vec,phi_f)
    end

    # Append new feature vector (for particular node) to the matrix
    feature_matrix = isempty(feature_matrix) ? feature_vec : hcat(feature_matrix,feature_vec)
  end
  return feature_matrix
end


function construct_prob_features(nodes, prob_data, prob_features)
  n_nodes = length(nodes)
  x_dim = 4

  X0, obs_data = prob_data[1:x_dim], prob_data[x_dim+1:end]
  n_obs = Int(length(obs_data) / 4)  # 4 coordinates per box

  # Ensure that "obstacles_map" at back if used
  if "obstacles_map" in prob_features 
    prob_features = push!(filter!(x->x≠"obstacles_map", prob_features), "obstacles_map")
  end

  feature_matrix = []
  for (idx, node) in enumerate(nodes)
    feature_vec = []
    for feature in prob_features
      if feature == "X0"
        phi_f = X0
      elseif feature == "obstacles"
        phi_f = obs_data
      elseif feature == "obstacles_map"
        # Constructs W1xH1 sized image with 1's where obstacles appear

        # Scale (x,y) table coordinates to image pixel (px,py)
        ft2m = 0.3048
        posmax = ft2m * [12.,9.]    # TODO(acauligi): don't hardcode free-flyer table size
        W1,H1 = 40, 30              # TODO(acauligi): don't hardcode image size

        img = zeros(Float64,H1,W1,n_obs,1)
        for ii_obs in 1:n_obs
          # TODO(pculbertson): check math lol
          obs = obs_data[4*(ii_obs-1)+1:4*ii_obs]
          col_range = Int(max(round(obs[1]/posmax[1]*W1),1)):Int(min(round(obs[2]/posmax[1]*W1),W1))
          row_range = Int(max(round(obs[3]/posmax[2]*H1),1)):Int(min(round(obs[4]/posmax[2]*H1),H1))

          img[row_range,col_range,ii_obs,:] .= 1.
        end
        phi_f = img[:]
      else
        # If feature not good
        println(string("Feature ",feature," is unknown."))
        continue
      end

      # Concatenate to the current
      feature_vec = isempty(feature_vec) ? phi_f : vcat(feature_vec,phi_f)
    end

    # Append new feature vector (for particular node) to the matrix
    feature_matrix = isempty(feature_matrix) ? feature_vec : hcat(feature_matrix,feature_vec)
  end
  return feature_matrix
end


function construct_features(nodes, bnb_prob, prob_data, bnb_features, prob_features)
  if !isempty(bnb_features) && !isempty(prob_features)
    bnb_feature_matrix = construct_bnb_features(nodes, bnb_prob, bnb_features)
    prob_feature_matrix = construct_prob_features(nodes, prob_data, prob_features)
    feature_matrix = vcat(bnb_feature_matrix, prob_feature_matrix)
  elseif !isempty(bnb_features)
    feature_matrix = construct_bnb_features(nodes, bnb_prob, bnb_features)
  elseif !isempty(prob_features)
    feature_matrix = construct_prob_features(nodes, prob_data, prob_features)
  else
    println("No features given.")
    return
  end
  return feature_matrix
end

function query_feature_size(problem_dict, bnb_features, prob_features)
  """ Input is the problem data dictionary that was saved """

  num_obstacles::Int64 = size(problem_dict["O"][1], 1)
  feature_size::Int64 = 0
  for feature in bnb_features
    if feature == "node_LB"
      feature_size += 1
    elseif feature == "node_assignments"
      feature_size += 4*(problem_dict["prob"]["N"]-1)*num_obstacles
    elseif feature == "depth"
      feature_size += 1
    else
      println(string("Feature ",feature," is uknown."))
    end
  end

  for feature in prob_features
    if feature == "X0"
      feature_size += 4
    elseif feature == "obstacles"
      feature_size += 4*num_obstacles
    elseif feature == "obstacles_map"
      W1,H1 = 40, 30              # TODO(acauligi): don't hardcode image size
      feature_size += W1*H1*num_obstacles
    else
      println(string("Feature ",feature," is uknown."))
    end
  end
  return feature_size
end


function getProblemData(dataset, prob_idx)
  """ Input the dataset (dictionary with X, O, Y, ...) and
  the index of the problem saved in that dataset. Outputs the vector
  that contains [x0; O] """
  x_dim,N = size(dataset["X"][prob_idx])
  n_obs = length(dataset["O"][prob_idx])
  obstacles = vcat([vcat(dataset["O"][prob_idx][i]...) for i in 1:n_obs]...)
  prob_data = vcat(dataset["X"][prob_idx][:,1], obstacles)

  Dict("prob_data"=>prob_data, "x_dim"=>x_dim, "N"=>N, "n_obs"=>n_obs)
end
