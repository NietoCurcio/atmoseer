import numpy as np
import torch
import gc
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import getopt
import time
import pickle

from numpy import array, mean
from typing import List
from math import cos, asin, sqrt
from utils.model import NetOrdinalClassification, label2ordinalencoding, NetRegression
from utils.training import fit, create_train_n_val_loaders, DeviceDataLoader, to_device, gen_learning_curve, seed_everything


def train(X_train, y_train, X_val, y_val, ordinal_regression, pipeline_id):
  N_EPOCHS = 5000
  PATIENCE = 500
  LEARNING_RATE = .3e-6
  NUM_FEATURES = X_train.shape[2]
  BATCH_SIZE = 512
  weight_decay = 1e-6

  if ordinal_regression:
    NUM_CLASSES = 5
    model = NetOrdinalClassification(in_channels=NUM_FEATURES,
                                     num_classes=NUM_CLASSES)
    y_train, y_val = label2ordinalencoding(y_train, y_val)
  else:
    global y_mean_value
    y_mean_value = np.mean(y_train)
    print(y_mean_value)
    model = NetRegression(in_channels=NUM_FEATURES, y_mean_value=y_mean_value)

  # model.apply(initialize_weights)

  criterion = nn.MSELoss()

  print(model)

  train_loader, val_loader = create_train_n_val_loaders(
      X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE)

  optimizer = torch.optim.Adam(model.parameters(),
                               lr=LEARNING_RATE,
                               weight_decay=weight_decay)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train_loader = DeviceDataLoader(train_loader, device)
  val_loader = DeviceDataLoader(val_loader, device)
  to_device(model, device)

  train_loss, val_loss = fit(model,
                             n_epochs=N_EPOCHS,
                             optimizer=optimizer,
                             train_loader=train_loader,
                             val_loader=val_loader,
                             patience=PATIENCE,
                             criterion=criterion,
                             pipeline_id=pipeline_id)

  gen_learning_curve(train_loss, val_loss, pipeline_id)

  return model


def main(argv):
    help_message = "Usage: {0} -p <pipeline_id>".format(argv[0])

    try:
        opts, args = getopt.getopt(
            argv[1:], "hp:r:", ["help", "pipeline_id=", "reg"])
    except:
        print(help_message)
        sys.exit(2)

    ordinal_regression = True
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)  # print the help message
            sys.exit(2)
        elif opt in ("-p", "--pipeline_id"):
            pipeline_id = arg
        elif opt in ("-r", "--reg"):
            ordinal_regression = False

    start_time = time.time()

    seed_everything()

    #
    # Load np arrays (stored in a pickle file) from disk
    #
    file = open("../data/datasets/" + pipeline_id + ".pickle", 'rb')
    (X_train, y_train, min_y_train, max_y_train,
        X_val, y_val, min_y_val, max_y_val,
        X_test, y_test, min_y_test, max_y_test) = pickle.load(file)

    #
    # Build model
    #
    model = train(X_train, y_train, X_val, y_val,
                  ordinal_regression, pipeline_id)

    # Load the best model
    model.load_state_dict(torch.load(
        '../models/best_' + pipeline_id + '.pt'))

    model.evaluate(X_test, y_test)

    print("--- %s seconds ---" % (time.time() - start_time))

# python train_model.py -p A652_E-N_EI
if __name__ == "__main__":
    main(sys.argv)
