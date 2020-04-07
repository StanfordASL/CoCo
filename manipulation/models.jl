using Flux: Chain, Dense, Conv, LayerNorm, relu, softmax, param

function get_linear_pruner(dense_length::Int;
  neurons::Int=32,x_dim::Int=2,N::Int=11,n_obs::Int=0)
  # 1=prune, 0=expand
  shape = [dense_length, 1]
  model = FFNet(shape).cuda()
  model
end

function get_ff_classifier(dense_length::Int,output_len::Int;
  neurons::Int=32,x_dim::Int=2,N::Int=11,n_obs::Int=0,depth::Int=3)
  ff_shape = [dense_length]
  for ii in 1:depth
    push!(ff_shape, neurons)
  end
  push!(ff_shape, output_len)
  # model = torch.nn.DataParallel(FFNet(ff_shape,activation=torch.nn.functional.relu).cuda())
  model = FFNet(ff_shape,activation=torch.nn.functional.relu)
  model
end

function get_ff_regressor(dense_length::Int,output_len::Int;
  neurons::Int=32,x_dim::Int=4,N::Int=11,n_obs::Int=1,depth::Int=3)
  ff_shape = [dense_length]
  for ii in 1:depth
    push!(ff_shape, neurons)
  end
  push!(ff_shape, output_len)
  # model = torch.nn.DataParallel(FFNet(ff_shape,activation=torch.nn.functional.relu).cuda())
  model = FFNet(ff_shape,activation=torch.nn.functional.relu)
  model
end
