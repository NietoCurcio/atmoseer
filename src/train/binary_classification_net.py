import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from train.training_utils import *
from train.evaluate import *
from rainfall_prediction import onehotencoding_to_binarylabels
from rainfall_classification_base import RainfallClassificationBase
from train.early_stopping import *


class BinaryClassificationNet(RainfallClassificationBase):
    def __init__(self, in_channels, num_classes):
        super(BinaryClassificationNet, self).__init__()
        self.conv1d_1 = nn.Conv1d(
            in_channels=in_channels, out_channels=32, kernel_size=3, padding=2)
        self.conv1d_2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, padding=2)
        self.conv1d_3 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, padding=2)
        self.conv1d_4 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=2)

        self.relu = nn.GELU()

        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(896, 50)

        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.relu(x)

        x = self.conv1d_2(x)
        x = self.relu(x)

        x = self.conv1d_3(x)
        x = self.relu(x)

        x = self.conv1d_4(x)

        x = self.relu(x)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        x = nn.functional.softmax(x, dim=1)

        return x

    def predict(self, X):
        print('Making predictions with ordinal classification model...')
        self.eval()

        test_x_tensor = torch.from_numpy(X.astype('float64'))
        test_x_tensor = torch.permute(test_x_tensor, (0, 2, 1))

        outputs = []
        with torch.no_grad():
            output = self(test_x_tensor.float())
            yb_pred_encoded = output.detach().cpu().numpy()
            yb_pred_decoded = onehotencoding_to_binarylabels(yb_pred_encoded)
            outputs.append(yb_pred_decoded.reshape(-1, 1))

        y_pred = np.vstack(outputs)

        return y_pred

    def evaluate(self, X_test, y_test):
        print('Evaluating ordinal classification model...')
        self.eval()

        test_x_tensor = torch.from_numpy(X_test.astype('float64'))
        test_x_tensor = torch.permute(test_x_tensor, (0, 2, 1))
        test_y_tensor = torch.from_numpy(y_test.astype('float64'))

        test_ds = TensorDataset(test_x_tensor, test_y_tensor)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=32, shuffle=False)
        test_loader = DeviceDataLoader(test_loader, get_default_device())

        # test_losses = []
        outputs = []
        with torch.no_grad():
            for xb, yb in test_loader:
                output = self(xb.float())
                yb_pred_encoded = output.detach().cpu().numpy()
                yb_pred_decoded = onehotencoding_to_binarylabels(
                    yb_pred_encoded)
                # print(yb_pred_decoded)
                outputs += yb_pred_decoded

        # print(outputs)
        y_pred = np.vstack(outputs)

        print(y_test)
        print(y_pred.shape)

        export_confusion_matrix_to_latex(y_test, y_pred)

    def fit(self, n_epochs, optimizer, train_loader, val_loader, patience, criterion, pipeline_id):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for epoch in range(n_epochs):

            ###################
            # train the model #
            ###################
            self.train()  # prep model for training
            for data, target in train_loader:
                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                output = self(data.float())

                # calculate the loss
                loss = criterion(output, target.float())
                assert not (np.isnan(loss.item()) or loss.item() >
                            1e6), f"Loss explosion: {loss.item()}"

                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()

                # record training loss
                train_losses.append(loss.item())

            ######################
            # validate the model #
            ######################
            self.eval()  # prep model for evaluation
            for data, target in val_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self(data.float())
                # calculate the loss
                loss = criterion(output, target.float())
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{(epoch+1):>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            early_stopping(valid_loss, self, pipeline_id)

            if early_stopping.early_stop:
                print("Early stopping activated!")
                break

        return avg_train_losses, avg_valid_losses
