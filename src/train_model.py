import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time

import numpy as np
import sys
import argparse
import time
import train.pipeline as pipeline

from train.ordinal_classifier import OrdinalClassificationNet
from train.binary_classifier import BinaryClassifier
from train.regression_net import Regressor
from train.training_utils import DeviceDataLoader, to_device, gen_learning_curve, seed_everything
from train.conv1d_neural_net import Conv1DNeuralNet 
from train.lstm_neural_net import LstmNeuralNet
import rainfall as rp

import logging

def compute_weights_for_binary_classification(y):
    '''
     TODO: really compute the weights!
    '''
    print(y.shape)
    weights = np.zeros_like(y)

    mask_NORAIN = np.where(y == 0)[0]
    print(mask_NORAIN.shape)

    mask_RAIN = np.where(y > 0)[0]
    print(mask_RAIN.shape)

    weights[mask_NORAIN] = 1
    weights[mask_RAIN] = 1

    print(f"# NO_RAIN records: {len(mask_NORAIN)}")
    print(f"# RAIN records: {len(mask_RAIN)}")
    return weights


def compute_weights_for_ordinal_classification(y):
    weights = np.zeros_like(y)

    mask_weak = np.logical_and(y >= 0, y < 5)
    weights[mask_weak] = 1

    mask_moderate = np.logical_and(y >= 5, y < 25)
    weights[mask_moderate] = 5

    mask_strong = np.logical_and(y >= 25, y < 50)
    weights[mask_strong] = 25

    mask_extreme = np.logical_and(y >= 50, y < 150)
    weights[mask_extreme] = 50

    return weights


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum()


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, pos_weight=None, neg_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)

        # Apply class weights to positive and negative examples
        if self.pos_weight is not None and self.neg_weight is not None:
            loss = self.neg_weight * \
                (1 - targets) * torch.log(1 - inputs) + \
                self.pos_weight * targets * torch.log(inputs)
        elif self.pos_weight is not None:
            loss = self.pos_weight * targets * torch.log(inputs)
        elif self.neg_weight is not None:
            loss = self.neg_weight * (1 - targets) * torch.log(1 - inputs)
        else:
            loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return -torch.mean(loss)


def train(X_train, y_train, X_val, y_val, prediction_task_id, pipeline_id, learner, config, resume_training: bool = False):
    NUM_FEATURES = X_train.shape[2]
    print(f"Number of features: {NUM_FEATURES}")

    # model.apply(initialize_weights)

    if prediction_task_id == rp.PredictionTask.ORDINAL_CLASSIFICATION:
        N_EPOCHS = config["training"]["oc"]["N_EPOCHS"]
        PATIENCE = config["training"]["oc"]["PATIENCE"]
        BATCH_SIZE = config["training"]["oc"]["BATCH_SIZE"]
        WEIGHT_DECAY = config["training"]["oc"]["WEIGHT_DECAY"]
        LEARNING_RATE = config["training"]["oc"]["LEARNING_RATE"]
        DROPOUT_RATE = config["training"]["oc"]["DROPOUT_RATE"]
        NUM_CLASSES = config["training"]["oc"]["NUM_CLASSES"]
        SEQ_LENGHT = config["preproc"]["SLIDING_WINDOW_SIZE"]

        train_weights = compute_weights_for_ordinal_classification(y_train)
        val_weights = compute_weights_for_ordinal_classification(y_val)
        train_weights = torch.FloatTensor(train_weights)
        val_weights = torch.FloatTensor(val_weights)
        loss = weighted_mse_loss

        y_train = rp.value_to_ordinal_encoding(y_train)
        y_val = rp.value_to_ordinal_encoding(y_val)

        target_average = torch.mean(torch.tensor(
            y_train, dtype=torch.float32), dim=0, keepdim=True)

        input_dim = (NUM_FEATURES, SEQ_LENGHT)
        predictor = OrdinalClassificationNet(in_channels=NUM_FEATURES,
                                              input_dim=input_dim,
                                              num_classes=NUM_CLASSES,
                                              target_average=target_average,
                                              dropout_rate=DROPOUT_RATE)

    elif prediction_task_id == rp.PredictionTask.BINARY_CLASSIFICATION:
        print("- Prediction task: binary classification.")
        train_weights = None
        val_weights = None

        N_EPOCHS = config["training"]["bc"]["N_EPOCHS"]
        PATIENCE = config["training"]["bc"]["PATIENCE"]
        BATCH_SIZE = config["training"]["bc"]["BATCH_SIZE"]
        WEIGHT_DECAY = config["training"]["bc"]["WEIGHT_DECAY"]
        LEARNING_RATE = config["training"]["bc"]["LEARNING_RATE"]
        DROPOUT_RATE = config["training"]["bc"]["DROPOUT_RATE"]
        SEQ_LENGHT = config["preproc"]["SLIDING_WINDOW_SIZE"]

        # weights = torch.FloatTensor([1.0, 5.0])
        loss = nn.BCELoss()
        # loss = WeightedBCELoss(pos_weight=2, neg_weight=1)

        y_train = rp.value_to_binary_level(y_train)
        y_val = rp.value_to_binary_level(y_val)

        predictor = BinaryClassifier(learner)
    elif prediction_task_id == rp.PredictionTask.REGRESSION:
        print("- Prediction task: regression.")
        loss = nn.MSELoss()
        global y_mean_value
        y_mean_value = np.mean(y_train)
        print(y_mean_value)
        predictor = Regressor(in_channels=NUM_FEATURES, y_mean_value=y_mean_value)

    print(predictor)

    optimizer = torch.optim.Adam(
        predictor.learner.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f" - Setting up optimizer: {optimizer}")
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    print(f" - Creating data loaders.")
    train_loader = learner.create_dataloader(
        X_train, y_train, batch_size=BATCH_SIZE, weights=train_weights)
    val_loader = learner.create_dataloader(
        X_val, y_val, batch_size=BATCH_SIZE, weights=train_weights)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" - Moving data and parameters to {device}.")
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    to_device(predictor.learner, device)

    # resume_training = True
    # if resume_training:
    #     model_path = globals.MODELS_DIR + "best_" + pipeline_id + ".pt"  # Path to the pretrained model file
    #     model.load_state_dict(torch.load(model_path))

    print(f" - Fitting model...", end=" ")
    train_loss, val_loss = predictor.learner.fit(n_epochs=N_EPOCHS,
                                          optimizer=optimizer,
                                          train_loader=train_loader,
                                          val_loader=val_loader,
                                          patience=PATIENCE,
                                          criterion=loss,
                                          pipeline_id=pipeline_id)
    print("Done!")

    gen_learning_curve(train_loss, val_loss, pipeline_id)

    return predictor


def main(argv):
    parser = argparse.ArgumentParser(description="Train a rainfall forecasting model.")
    parser.add_argument("-t", "--task", choices=["ORDINAL_CLASSIFICATION", "BINARY_CLASSIFICATION"],
                        default="REGRESSION", help="Prediction task")
    parser.add_argument("-l", "--learner", choices=["Conv1DNeuralNet", "LstmNeuralNet"],
                        default="LstmNeuralNet", help="Learning algorithm to be used.")
    parser.add_argument("-p", "--pipeline_id", required=True, help="Pipeline ID")
    
    args = parser.parse_args(argv[1:])

    prediction_task_id = None

    prediction_task_id = None

    if args.task == "ORDINAL_CLASSIFICATION":
        prediction_task_id = rp.PredictionTask.ORDINAL_CLASSIFICATION
    elif args.task == "BINARY_CLASSIFICATION":
        prediction_task_id = rp.PredictionTask.BINARY_CLASSIFICATION

    seed_everything()

    X_train, y_train, X_val, y_val, X_test, y_test = pipeline.load_datasets(
        args.pipeline_id)

    if prediction_task_id == rp.PredictionTask.ORDINAL_CLASSIFICATION:
        args.pipeline_id += "_OC"
    elif prediction_task_id == rp.PredictionTask.BINARY_CLASSIFICATION:
        args.pipeline_id += "_BC"
    elif prediction_task_id == rp.PredictionTask.REGRESSION:
        args.pipeline_id += "_Reg"

    # Use globals() to access the global namespace and find the class by name
    class_name = args.learner
    print(class_name)
    class_obj = globals()[class_name]

    # Check if the class exists
    if not isinstance(class_obj, type):
        raise ValueError(f"Class '{class_name}' not found.")

    args.pipeline_id += "_" + class_name

    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    DROPOUT_RATE = config["training"]["bc"]["DROPOUT_RATE"]
    BATCH_SIZE = config["training"]["bc"]["BATCH_SIZE"]
    SEQ_LENGTH = config["preproc"]["SLIDING_WINDOW_SIZE"]

    NUM_FEATURES = X_train.shape[2]
    print(f"Number of features: {NUM_FEATURES}")

    # Instantiate the class
    learner = class_obj(seq_length=SEQ_LENGTH,
                        input_size=NUM_FEATURES, dropout_rate=DROPOUT_RATE)
    print(f'Learner: {learner}')

    # Build model
    start_time = time.time()
    model = train(X_train, y_train, X_val, y_val, prediction_task_id, args.pipeline_id, learner, config)
    logging.info("Model training took %s seconds." % (time.time() - start_time))

    # Evaluate using the best model produced
    test_loader = learner.create_dataloader(X_test, y_test, batch_size=BATCH_SIZE)
    model.print_evaluation_report(args.pipeline_id, test_loader)

if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    execution_time = end_time - start_time
    print("The execution time was", execution_time, "seconds.")
