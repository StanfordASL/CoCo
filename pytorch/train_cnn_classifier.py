import os
import pdb
import sys
import time 
import h5py
import random
import string
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models import FFNet, BnBCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(fn_dataset):
  fn_pt_model = fn_dataset.split(".jld")[0] + "_cnn_classifier.pt"
  print('FN: {}'.format(fn_pt_model))

  writer = SummaryWriter()

  # Load dataset
  f = h5py.File(fn_dataset, "r")
  X = f['X'].value
  Y = f['Y'].value[:,0]
  Y = np.subtract(Y, 1)   # convert 1-indexing to 0-indexing

  print('X_dim: {} x {}'.format(X.shape[0], X.shape[1]))
  print('Y_dim: {}'.format(Y.shape[0]))
  N_strategies = f['N_strategies'].value
  feature_size = f['feature_size'].value
  print('Feature size: {}'.format(feature_size))
  print('N_strategies: {}'.format(N_strategies))

  channels = f['channels'].value
  n_obs = channels[0]
  W = f['W'].value
  H = f['H'].value
  input_size = (H,W)
 
  ff_feature_len = X.shape[1]-H*W*n_obs

  prob_features = f['prob_features'].value
  if 'obstacles_map' not in prob_features:
    print('Map of obstacles not in feature vector! Exiting training')
    return

  # Load model from file name
  depth   = int(fn_dataset.split("_")[-2].split("-")[0])
  neurons = int(fn_dataset.split("_")[-1].split("-")[0])
  print('Creating {} layer network with {} neurons'.format(depth,neurons))

  ff_shape = []
  for ii in range(depth):
    ff_shape.append(neurons)
  ff_shape.append(N_strategies)

  model = BnBCNN(feature_size, channels, ff_shape, input_size,
    conv_activation=torch.nn.ReLU(), ff_activation=torch.nn.ReLU()).cuda()

  if os.path.exists(fn_pt_model):
    print('Loading presaved model from {}'.format(fn_pt_model))
    model.load_state_dict(torch.load(fn_pt_model))

  # training parameters
  TRAINING_ITERATIONS = int(150)
  BATCH_SIZE = 128
  CHECKPOINT_AFTER = int(1000)
  SAVEPOINT_AFTER = int(30000)
  TEST_BATCH_SIZE = 32000 #2048 maps / batch 32

  training_loss = torch.nn.CrossEntropyLoss()
  opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.00001)

  indices = list(range(0,N_strategies))
  random.shuffle(indices)

  itr = 1
  for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
    t0 = time.time()
    running_loss = 0.0
    rand_idx = list(np.arange(0,X.shape[0]-1))
    random.shuffle(rand_idx)

    # Sample all data points
    indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

    for ii,idx in enumerate(indices):
      # zero the parameter gradients
      opt.zero_grad()
      img_batch = Variable(torch.from_numpy(X[idx,ff_feature_len:].reshape((BATCH_SIZE,n_obs,H,W)))).float().cuda()
      feature_batch = Variable(torch.from_numpy(X[idx,0:ff_feature_len])).float().cuda()
      labels = Variable(torch.from_numpy(Y[idx])).long().cuda()

      # forward + backward + optimize
      outputs = model(img_batch, feature_batch)
      loss = training_loss(outputs, labels).float().cuda()
      class_guesses = torch.argmax(outputs,1)
      accuracy = torch.mean(torch.eq(class_guesses,labels).float())
      loss.backward()
      #torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
      opt.step()

      # print statistics\n",
      running_loss += loss.item()
      if itr % CHECKPOINT_AFTER == 0:
        #print('[%d] loss: %.3f' %
          #(epoch + 1, running_loss / SAVEPOINT_AFTER))
        #running_loss = 0.0
        rand_idx = list(np.arange(0,X.shape[0]-1))
        random.shuffle(rand_idx)
        test_inds = rand_idx[:TEST_BATCH_SIZE]

        img_batch = Variable(torch.from_numpy(X[test_inds,ff_feature_len:].reshape((BATCH_SIZE,H,W,n_obs)))).float().cuda()
        feature_batch = Variable(torch.from_numpy(X[test_inds,0:ff_feature_len])).float().cuda()
        labels = Variable(torch.from_numpy(Y[test_inds])).long().cuda()

        # forward + backward + optimize
        outputs = model(img_batch, feature_batch)
        loss = training_loss(outputs, labels).float().cuda()
        class_guesses = torch.argmax(outputs,1)
        accuracy = torch.mean(torch.eq(class_guesses,labels).float())
        print("loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))
        
        
        # for jj in range(TEST_BATCHES):
          # DO SOMETHING

      if itr % SAVEPOINT_AFTER == 0:
        torch.save(model.state_dict(), fn_pt_model)
        print('Saved model at {}'.format(fn_pt_model))
        writer.add_scalar('Loss/train', running_loss, epoch)

      itr += 1

    print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

  torch.save(model.state_dict(), fn_pt_model)
  print('Saved model at {}'.format(fn_pt_model))

  print('Done training')

if __name__=='__main__':
  main(sys.argv[1])
