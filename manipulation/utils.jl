using LinearAlgebra

function vectorizeIntegerData(yArray)
  """ Takes the data from dictionary dict[-]["Y"][-] which is a
  (4,N) array and convert it into the correct vector form (column wise
  stacking) and casts as Ints. """
  yVector = yArray'[:]
  return Int.(round.(yVector))
end

function construct_bnb_features(nodes, bnb_prob, bnb_features)
  feature_matrix = []
  feature_matrix
end

function construct_prob_features(nodes, prob_data, prob_features)
  feature_matrix = []

  h,r,mu = prob_data[1:3]
  w = prob_data[4:end]

  for (idx, node) in enumerate(nodes)
    feature_vec = []
    for feature in prob_features
      if feature == "h"
        phi_f = h
      elseif feature == "r"
        phi_f = r
      elseif feature == "mu"
        phi_f = mu
      elseif feature == "ws"
        phi_f = w
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

  feature_matrix
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
  end

  for feature in prob_features
    if feature == "h"
      feature_size += 1
    elseif feature == "r"
      feature_size += 1
    elseif feature == "mu"
      feature_size += 1
    elseif feature == "ws"
      feature_size += 12 
    else
      println(string("Feature ",feature," is uknown."))
    end
  end

  feature_size
end

function getProblemData(dataset, prob_idx)
  """ Input the dataset (dictionary with X, Y, ...) and
  the index of the problem saved in that dataset. Outputs the vector
  that contains [X0;Xg] """
  prob_data = vcat(dataset["h"][prob_idx],
                  dataset["r"][prob_idx],
                  dataset["mu"][prob_idx],
                  dataset["ws"][prob_idx])

  Dict("prob_data"=>prob_data)
end
