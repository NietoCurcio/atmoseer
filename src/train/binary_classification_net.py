import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from train.training_utils import *
from train.evaluate import *
from rainfall_prediction import onehotencoding_to_binarylabels
from rainfall_classification_base import RainfallClassificationBase

class BinaryClassificationNet(RainfallClassificationBase):
    def __init__(self, in_channels, num_classes):
        super(BinaryClassificationNet,self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels = in_channels, out_channels = 32, kernel_size = 3, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=2)
        self.conv1d_3 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, padding=2)
        self.conv1d_4 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding=2)

        self.relu = nn.GELU()
        
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(896,50)

        self.fc2 = nn.Linear(50, num_classes)


    def forward(self,x):
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
        x = self.softmax(x)

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
          yb_pred_decoded = onehotencoding_to_binarylabels(yb_pred_encoded)
          outputs.append(yb_pred_decoded.reshape(-1,1))

      y_pred = np.vstack(outputs)

      export_confusion_matrix_to_latex(y_test, y_pred)

def fit(self, epochs, lr, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(self.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        self.train()
        train_losses = []
        for batch in train_loader:
            loss = self.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = self.evaluate(self, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        self.epoch_end(epoch, result)
        history.append(result)
    return history