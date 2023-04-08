import torch
import numpy as np
import sys
import getopt
import time
import pickle

from train.ordinal_classification_net import OrdinalClassificationNet, label2ordinalencoding
from train.regression_net import RegressionNet
from train.training_utils import fit, create_train_n_val_loaders, DeviceDataLoader, to_device, gen_learning_curve, seed_everything

def compute_weights(input_array):
    output_array = np.zeros_like(input_array)

    mask_weak = np.logical_and(input_array >= 0, input_array < 5)
    output_array[mask_weak] = 1

    mask_moderate = np.logical_and(input_array >= 5, input_array < 25)
    output_array[mask_moderate] = 5

    mask_strong = np.logical_and(input_array >= 25, input_array < 50)
    output_array[mask_strong] = 25

    mask_extreme = np.logical_and(input_array >= 50, input_array < 150)
    output_array[mask_extreme] = 50

    return output_array

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum()

def train(X_train, y_train, X_val, y_val, ordinal_regression, pipeline_id):
    N_EPOCHS = 20000
    PATIENCE = 500
    LEARNING_RATE = .3e-6
    NUM_FEATURES = X_train.shape[2]
    BATCH_SIZE = 1024
    weight_decay = 1e-6

    # model.apply(initialize_weights)
    train_weights = compute_weights(y_train)
    val_weights = compute_weights(y_val)
    train_weights = torch.FloatTensor(train_weights)
    val_weights = torch.FloatTensor(val_weights)
    # print(weights.shape)
    # print(weights[:5])
    # criterion = nn.MSELoss()
    criterion = weighted_mse_loss

    if ordinal_regression:
        NUM_CLASSES = 5
        model = OrdinalClassificationNet(in_channels=NUM_FEATURES,
                                         num_classes=NUM_CLASSES)
        y_train, y_val = label2ordinalencoding(y_train, y_val)
    else:
        global y_mean_value
        y_mean_value = np.mean(y_train)
        print(y_mean_value)
        model = RegressionNet(in_channels=NUM_FEATURES,
                              y_mean_value=y_mean_value)

    print(model)

    train_loader, val_loader = create_train_n_val_loaders(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, train_weights=train_weights, val_weights=val_weights)

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
            argv[1:], "hp:r", ["help", "pipeline_id=", "reg"])
    except:
        print(help_message)
        sys.exit(2)

    ordinal_classification = True

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_message)  # print the help message
            sys.exit(2)
        elif opt in ("-p", "--pipeline_id"):
            pipeline_id = arg
        elif opt in ("-r", "--reg"):
            ordinal_classification = False

    start_time = time.time()

    seed_everything()

    #
    # Load np arrays (stored in a pickle file) from disk
    #
    file = open("../data/datasets/" + pipeline_id + ".pickle", 'rb')
    (X_train, y_train,  # min_y_train, max_y_train,
        X_val, y_val,  # min_y_val, max_y_val,
        X_test, y_test) = pickle.load(file)

    if ordinal_classification:
        pipeline_id += "_OC" 
    else:
        pipeline_id += "_Reg" 

    #
    # Build model
    #
    model = train(X_train, y_train, X_val, y_val,
                  ordinal_classification, pipeline_id)

    # Load the best model
    model.load_state_dict(torch.load(
        '../models/best_' + pipeline_id + '.pt'))

    model.evaluate(X_test, y_test)

    print("--- %s seconds ---" % (time.time() - start_time))


# python train_model.py -p A652_E-N_EI
if __name__ == "__main__":
    main(sys.argv)
