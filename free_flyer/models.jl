using Flux: Chain, Dense, Conv, LayerNorm, relu, softmax, param

function get_cnn_model(dense_length::Int;
  neurons::Int=64,x_dim::Int=4,N::Int=6,n_obs::Int=4,
  fltr::Int=2,str::Int=2,pd::Int=1,n_channels::Int=8,
  depth::Int=3)
  W1,H1 = 30,40               # TODO(acauligi): don't hardcode image size
  input_size = (W1,H1)
  ff_shape = []
  for ii in 1:depth
    push!(ff_shape, neurons)
  end
  push!(ff_shape, 1)
  num_ff_features = dense_length - W1*H1*n_obs

  channels = [n_obs,8,8]

  model = BnBCNN(num_ff_features,channels,ff_shape,input_size).cuda()
end

function get_ff_classifier(dense_length::Int,output_len::Int;
  neurons::Int=32,x_dim::Int=4,N::Int=6,n_obs::Int=1,depth::Int=3)
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
  neurons::Int=32,x_dim::Int=4,N::Int=6,n_obs::Int=1,depth::Int=3)
  ff_shape = [dense_length]
  for ii in 1:depth
    push!(ff_shape, neurons)
  end
  push!(ff_shape, output_len)
  # model = torch.nn.DataParallel(FFNet(ff_shape,activation=torch.nn.functional.relu).cuda())
  model = FFNet(ff_shape,activation=torch.nn.functional.relu)
  model
end
