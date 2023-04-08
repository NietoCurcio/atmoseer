"""
    https://stats.stackexchange.com/questions/209290/deep-learning-for-ordinal-classification

    https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c

    https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99

    https://arxiv.org/pdf/0704.1028.pdf

    https://datascience.stackexchange.com/questions/44354/ordinal-classification-with-xgboost

    https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

    https://neptune.ai/blog/how-to-deal-with-imbalanced-classification-and-regression-data

    https://colab.research.google.com/github/YyzHarry/imbalanced-regression/blob/master/tutorial/tutorial.ipynb#scrollTo=tSrzhog1gxyY
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from train.training_utils import *
from train.evaluation import *

NO_RAIN = 0
WEAK_RAIN = 1
MODERATE_RAIN = 2
STRONG_RAIN = 3
EXTREME_RAIN = 4

import numpy as np
def f(x):
  if x == NO_RAIN:
    return np.array([1,0,0,0,0])
  elif x == WEAK_RAIN:
    return np.array([1,1,0,0,0])
  elif x == MODERATE_RAIN:
    return np.array([1,1,1,0,0])
  elif x == STRONG_RAIN:
    return np.array([1,1,1,1,0])
  elif x == EXTREME_RAIN:
    return np.array([1,1,1,1,1])

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def ordinalencoding2labels(pred: np.ndarray):
    """
    Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

def label2ordinalencoding(y_train, y_val):
  no_rain_train, weak_rain_train, moderate_rain_train, strong_rain_train, extreme_rain_train = get_events_per_precipitation_level(y_train)
  no_rain_val, weak_rain_val, moderate_rain_val, strong_rain_val, extreme_rain_val = get_events_per_precipitation_level(y_val)

  y_train_class = np.zeros_like(y_train)
  y_val_class = np.zeros_like(y_val)

  y_train_class[no_rain_train] = NO_RAIN
  y_train_class[weak_rain_train] = WEAK_RAIN
  y_train_class[moderate_rain_train] = MODERATE_RAIN
  y_train_class[strong_rain_train] = STRONG_RAIN
  y_train_class[extreme_rain_train] = EXTREME_RAIN

  y_val_class[no_rain_val] = NO_RAIN
  y_val_class[weak_rain_val] = WEAK_RAIN
  y_val_class[moderate_rain_val] = MODERATE_RAIN
  y_val_class[strong_rain_val] = STRONG_RAIN
  y_val_class[extreme_rain_val] = EXTREME_RAIN

  y_train = np.array(list(map(f, y_train_class)))
  y_val = np.array(list(map(f, y_val_class)))

  return y_train, y_val


class OrdinalClassificationNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OrdinalClassificationNet,self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels = in_channels, out_channels = 32, kernel_size = 3, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=2)
        self.conv1d_3 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=2)
        self.conv1d_4 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding=2)

        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
        
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(896,50)

        self.fc2 = nn.Linear(50, num_classes)


    def forward(self,x):
        x = self.conv1d_1(x)
        x = self.relu(x)

        # x = self.max_pooling1d_1(x)

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
        x = self.sigmoid(x)

        return x

    def prediction2label(pred: np.ndarray):
      """Convert ordinal predictions to class labels, e.g.
      
      [0.9, 0.1, 0.1, 0.1] -> 0
      [0.9, 0.9, 0.1, 0.1] -> 1
      [0.9, 0.9, 0.9, 0.1] -> 2
      etc.
      """
      return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

    def training_step(self, batch):
        X_train, y_train = batch 
        out = self(X_train)                  # Generate predictions
        loss = F.cross_entropy(out, y_train) # Calculate loss
        return loss

    def validation_step(self, batch):
        X_train, y_train = batch 
        out = self(X_train)                    # Generate predictions
        loss = F.cross_entropy(out, y_train)   # Calculate loss
        acc = accuracy(out, y_train)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    def predict(self, X):
      print('Evaluating ordinal classification model...')
      self.eval()

      test_x_tensor = torch.from_numpy(X.astype('float64'))
      test_x_tensor = torch.permute(test_x_tensor, (0, 2, 1))

      outputs = []
      with torch.no_grad():
          output = self(test_x_tensor.float())
          yb_pred_encoded = output.detach().cpu().numpy()
          yb_pred_decoded = ordinalencoding2labels(yb_pred_encoded)
          outputs.append(yb_pred_decoded.reshape(-1,1))

      y_pred = np.vstack(outputs)

      return y_pred
       
    def evaluate(self, X_test, y_test):
      print('Evaluating ordinal classification model...')
      self.eval()

      test_x_tensor = torch.from_numpy(X_test.astype('float64'))
      test_x_tensor = torch.permute(test_x_tensor, (0, 2, 1))
      test_y_tensor = torch.from_numpy(y_test.astype('float64'))  

      test_ds = TensorDataset(test_x_tensor, test_y_tensor)
      test_loader = torch.utils.data.DataLoader(test_ds, batch_size = 32, shuffle = False)
      test_loader = DeviceDataLoader(test_loader, get_default_device())

      test_losses = []
      outputs = []
      with torch.no_grad():
        for xb, yb in test_loader:
          output = self(xb.float())
          yb_pred_encoded = output.detach().cpu().numpy()
          yb_pred_decoded = ordinalencoding2labels(yb_pred_encoded)
          outputs.append(yb_pred_decoded.reshape(-1,1))

      y_pred = np.vstack(outputs)

      export_confusion_matrix_to_latex(y_test, y_pred)
