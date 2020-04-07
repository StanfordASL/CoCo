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
from torch.nn import Sigmoid
import numpy as np

from models import FFNet, BnBCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(fn_dataset):
  fn_pt_model = fn_dataset.split(".jld")[0] + "_regressor.pt"
  print('FN: {}'.format(fn_pt_model))

  writer = SummaryWriter()

  # Load dataset
  f = h5py.File(fn_dataset, "r")
  X = f['X'].value
  Y = f['Y'].value[:,1:]
  n_y = Y.shape[1]

  N_strategies = f['N_strategies'].value
  feature_size = f['feature_size'].value

  print('Feature size: {}'.format(feature_size))
  print('N_strategies: {}'.format(N_strategies))
  print('X_dim: {} x {}'.format(X.shape[0], X.shape[1]))
  print('Y_dim: {} x {}'.format(Y.shape[0], Y.shape[1]))

  # Load model from file name
  depth   = int(fn_dataset.split("_")[-2].split("-")[0])
  neurons = int(fn_dataset.split("_")[-1].split("-")[0])
  print('Creating {} layer network with {} neurons'.format(depth,neurons))

  ff_shape = [feature_size]
  for ii in range(depth):
    ff_shape.append(neurons)
  ff_shape.append(n_y)

  model = FFNet(ff_shape, activation=torch.nn.ReLU()).cuda()

  if os.path.exists(fn_pt_model):
    print('Loading presaved model from {}'.format(fn_pt_model))
    model.load_state_dict(torch.load(fn_pt_model))

  # training parameters
  TRAINING_ITERATIONS = int(1500)
  BATCH_SIZE = 32
  CHECKPOINT_AFTER = int(2000)
  SAVEPOINT_AFTER = int(5000)
  TEST_BATCH_SIZE = 320 #2048 maps / batch 32

  training_loss = torch.nn.BCEWithLogitsLoss() # See: https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/45
  opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
  opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

  itr = 1
  for epoch in range(TRAINING_ITERATIONS):  # loop over the dataset multiple times
    t0 = time.time()
    running_loss = 0.0
    rand_idx = np.arange(0,X.shape[0]-1)
    random.shuffle(rand_idx)
    indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

    for ii,idx in enumerate(indices):
      # zero the parameter gradients
      opt.zero_grad()

      inputs = Variable(torch.from_numpy(X[idx,:])).float().cuda()
      y_true = Variable(torch.from_numpy(Y[idx,:])).float().cuda()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = training_loss(outputs, y_true).float().cuda()
      loss.backward()
      opt.step()

      # print statistics\n",
      running_loss += loss.item()
      if itr % CHECKPOINT_AFTER == 0:
        rand_idx = list(np.arange(0,X.shape[0]-1))
        random.shuffle(rand_idx)
        test_inds = rand_idx[:TEST_BATCH_SIZE]
        inputs = Variable(torch.from_numpy(X[test_inds,:])).float().cuda()
        y_out = Variable(torch.from_numpy(Y[test_inds])).float().cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = training_loss(outputs, y_out).float().cuda()
        outputs = Sigmoid()(outputs).round()
        accuracy = [float(all(torch.eq(outputs[ii],y_out[ii]))) for ii in range(TEST_BATCH_SIZE)]
        accuracy = np.mean(accuracy)
        print("loss:   "+str(loss.item()) + " , acc: " + str(accuracy))

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
