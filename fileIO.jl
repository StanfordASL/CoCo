using JLD

function readAllData(filename)
  """ Returns dictionary of data """
  data = load(filename)
  return data
end

function writeTrainingData(filename, training_data)
  """ Overwrites existing file """
  jldopen(filename, "w") do file
    for (k,v) in training_data
      write(file,k,v)
    end
  end
end

function writeEvalData(filename, num_LPs, fathom_count, time, dataset, prob_idx)
  """ Overwrites existing file """
  save(filename, "LPs_solved", num_LPs, "fathom_count", fathom_count, "solve_time", time, 
        "dataset", dataset, "prob_idx", prob_idx) 
end
