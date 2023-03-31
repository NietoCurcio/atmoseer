import numpy as np
import torch
import gc
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import getopt
import time

from numpy import array, mean
from typing import List
from math import cos, asin, sqrt
from utils.model import NetOrdinalClassification, label2ordinalencoding, NetRegression
from utils.training import fit, create_train_n_val_loaders, DeviceDataLoader, to_device, gen_learning_curve, seed_everything

cor_est = ['alto_da_boa_vista','guaratiba','iraja','jardim_botanico','riocentro','santa_cruz','sao_cristovao','vidigal']

def train(X_train, y_train, X_val, y_val, ordinal_regression):
  N_EPOCHS = 5000
  PATIENCE = 500
  LEARNING_RATE = .3e-6
  NUM_FEATURES = X_train.shape[2]
  BATCH_SIZE = 512
  weight_decay = 1e-6

  if ordinal_regression:
    NUM_CLASSES = 5
    model = NetOrdinalClassification(in_channels = NUM_FEATURES,
                                    num_classes = NUM_CLASSES)
    y_train, y_val = label2ordinalencoding(y_train, y_val)
  else:
    global y_mean_value
    y_mean_value = np.mean(y_train)
    print(y_mean_value)
    model = NetRegression(in_channels = NUM_FEATURES, y_mean_value = y_mean_value)

  # model.apply(initialize_weights)

  criterion = nn.MSELoss()

  print(model)

  train_loader, val_loader = create_train_n_val_loaders(X_train, y_train, X_val, y_val, batch_size = BATCH_SIZE)

  optimizer = torch.optim.Adam(model.parameters(), 
                               lr=LEARNING_RATE, 
                               weight_decay=weight_decay)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train_loader = DeviceDataLoader(train_loader, device)
  val_loader = DeviceDataLoader(val_loader, device)    
  to_device(model, device)
    
  train_loss, val_loss = fit(model, 
                             n_epochs = N_EPOCHS, 
                             optimizer = optimizer, 
                             train_loader = train_loader, 
                             val_loader = val_loader, 
                             patience = PATIENCE, 
                             criterion = criterion,
                             aux_nome = aux_nome,
                             num_sta = num_sta)

  gen_learning_curve(train_loss, val_loss,aux_nome,num_sta)

  return model

def build_model(ordinal_regression = True, file = ''):
  seed_everything()

  # TODO: get np arrays from disk

  model = train(X_train, y_train, X_val, y_val, ordinal_regression)
  
  # load the best model
  model.load_state_dict(torch.load('../model/Modelo_'+ aux_nome +'.pt'))

  model.evaluate(X_test, y_test)


def main(argv):
    arg_file = ""
    arg_model = True
    arg_help = "{0} -f <file> -s <sta>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hf:r:", ["help", "file=","reg="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--file"):
            arg_file = arg
        elif opt in ("-r", "--reg"):
            arg_model = False

    global aux_nome
    global num_sta
    aux_nome = ''
    num_sta = ''
    aux_nome = arg_file
    num_sta = str(arg_model)
    start_time = time.time()
    build_model(ordinal_regression = arg_model, file = arg_file)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main(sys.argv)

