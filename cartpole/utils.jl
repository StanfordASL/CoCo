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

  X0, Xg = prob_data[1:x_dim], prob_data[x_dim+1:2*x_dim]

  # TODO(acauligi): currently hard coded, read out of param file
  l = 1.
  mc = 1.
  mp = 1.
  g = 10.
  dh = 0.05
  dist = 0.5

  feature_matrix = []
  for (idx, node) in enumerate(nodes)
    feature_vec = []
    for feature in prob_features
      if feature == "X0"
        phi_f = X0
      elseif feature == "Xg"
        phi_f = Xg
      elseif feature == "delta2_0"
        phi_f = -X0[1] + l*X0[2] - dist
      elseif feature == "delta3_0"
        phi_f = X0[1] - l*X0[2] - dist
      elseif feature == "delta2_g"
        phi_f = -Xg[1] + l*Xg[2] - dist
      elseif feature == "delta3_g"
        phi_f = Xg[1] - l*Xg[2] - dist
      elseif feature == "dist_to_goal"
        phi_f = norm(Xg-X0,2)
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

  feature_size::Int64 = 0
  for feature in bnb_features
    if feature == "node_LB"
      feature_size += 1
    elseif feature == "node_assignments"
      feature_size += 4*(problem_dict["prob"]["N"]-1)
    elseif feature == "depth"
      feature_size += 1
    else
      println(string("Feature ",feature," is uknown."))
    end
  end

  for feature in prob_features
    if feature == "X0"
      feature_size += 4
    elseif feature == "Xg"
      feature_size += 4
    elseif feature == "delta2_0"
      feature_size += 1
    elseif feature == "delta3_0"
      feature_size += 1
    elseif feature == "delta2_g"
      feature_size += 1
    elseif feature == "delta3_g"
      feature_size += 1
    elseif feature == "dist_to_goal"
      feature_size += 1
    else
      println(string("Feature ",feature," is uknown."))
    end
  end
  return feature_size
end


function getProblemData(dataset, prob_idx)
  """ Input the dataset (dictionary with X, Y, ...) and
  the index of the problem saved in that dataset. Outputs the vector
  that contains [X0;Xg] """
  x_dim,N = size(dataset["X"][prob_idx])
  prob_data = vcat(dataset["X0"][prob_idx],dataset["Xg"][prob_idx])

  Dict("prob_data"=>prob_data, "x_dim"=>x_dim, "N"=>N)
end
