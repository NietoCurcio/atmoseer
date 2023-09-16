import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import functools
import operator
from train.early_stopping import EarlyStopping
import numpy as np

class BaseNeuralNet(nn.Module):
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

    def forward(self, x):
        # print(f'x.shape = {x.shape}')
        # print(f'x[0] = {x[0]}')
        out = self.feature_extractor(x)
        # print(f'out.shape = {out.shape}')
        out = out.reshape(out.shape[0], -1)
        # print(f'out.shape = {out.shape}')
        out = self.classifier(out)
        return out


class Conv1DNeuralNet(BaseNeuralNet):
    def __init__(self, seq_length, input_size, dropout_rate=0.5):
        super(BaseNeuralNet, self).__init__()
        self.seq_length = seq_length
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.feature_extractor = self._get_feature_extractor()
        self.classifier = self._get_classifier()
    
    def _get_feature_extractor(self):
        fe = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=16, kernel_size=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=self.dropout_rate)
        )
        # https://datascience.stackexchange.com/questions/40906/determining-size-of-fc-layer-after-conv-layer-in-pytorch
        input_dim = (self.input_size, self.seq_length)
        self.num_features_before_fcnn = functools.reduce(operator.mul, list(fe(torch.rand(1, *input_dim)).shape))
        print(f"num_features_before_fcnn = {self.num_features_before_fcnn}")
        return fe

    def _get_classifier(self):
        return nn.Sequential(
            nn.Linear(in_features=self.num_features_before_fcnn, out_features=50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def create_dataloader(self, X, y, batch_size, weights = None):
        '''
        The X parameter is a numpy array having the following shape:
                    [batch_size, input_size, sequence_len] 

        The nn.Conv1D module expects inputs having the following shape: 
                    [batch_size, sequence_len, input_size] 
        See https://stackoverflow.com/questions/62372938/understanding-input-shape-to-pytorch-conv1d
        '''
        X = torch.from_numpy(X.astype('float64'))
        X = torch.permute(X, (0, 2, 1))
        y = torch.from_numpy(y.astype('float64'))

        if weights is None:
            ds = TensorDataset(X, y)
        else:
            ds = TensorDataset(X, y, weights)

        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        return loader


# Needed because LSTM() returns tuple of (tensor, (recurrent state))
class extract_tensor(nn.Module):
    def forward(self,x):
        out, _ = x
        return out

class LstmNeuralNet(BaseNeuralNet):
    def __init__(self, seq_length, input_size, dropout_rate=0.5):
        super(BaseNeuralNet, self).__init__()
        self.seq_lenght = seq_length
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.feature_extractor = self._get_feature_extractor()
        self.classifier = self._get_classifier()

    def _get_feature_extractor(self):
        hidden_size = 32
        fe = nn.Sequential(
            nn.LSTM(input_size = self.input_size, hidden_size = hidden_size, batch_first = True),
            extract_tensor(),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.num_features_before_fcnn = hidden_size * self.seq_lenght
        return fe

    def _get_classifier(self):
        return nn.Sequential(
            nn.Linear(in_features=self.num_features_before_fcnn, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid()
         )

    def create_dataloader(self, X, y, batch_size, weights = None):
        '''
        The X parameter is a numpy array having the following shape:
                    [batch_size, sequence_len, input_size] 

        The nn.LSTM module (with 'batch_first = True') expects inputs having the following shape: 
                    [batch_size, sequence_len, input_size] 

        where:
            - sequence_length = number of timestamps
            - batch_size = size of each batch
            - input_size = the length of the vector describing each feature observed at each timestamp.

        See https://discuss.pytorch.org/t/using-lstm-after-conv1d-for-time-series-data/111140
        '''
        print(X.shape)
        print(y.shape)
        
        X = torch.from_numpy(X.astype('float64'))
        y = torch.from_numpy(y.astype('float64'))

        if weights is None:
            ds = TensorDataset(X, y)
        else:
            ds = TensorDataset(X, y, weights)

        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        return loader
