import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from train.training_utils import *
from train.evaluate import *
from train.base_classifier import BaseClassifier
import rainfall as rp
import yaml

class BinaryClassifier(BaseClassifier):
    def __init__(self, learner):#, input_size, input_dim, dropout_rate=0.5):
        super(BinaryClassifier, self).__init__()
        self.learner = learner

    def print_evaluation_report(self, pipeline_id, test_loader):
        self.learner.load_state_dict(torch.load(globals.MODELS_DIR + '/best_' + pipeline_id + '.pt'))
        
        print("\\begin{verbatim}")
        print(f"***Evaluation report for pipeline {pipeline_id}***")
        print("\\end{verbatim}")

        print("\\begin{verbatim}")
        print("***Hyperparameters***")
        with open('./config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        model_config = config['training']['bc']
        pretty_model_config = yaml.dump(model_config, indent=4)
        print(pretty_model_config)
        print("\\end{verbatim}")
        
        print("\\begin{verbatim}")
        print("***Model architecture***")
        print(self)
        print("\\end{verbatim}")

        print("\\begin{verbatim}")
        print('***Confusion matrix***')
        print("\\end{verbatim}")
        y_pred, y_test = self.evaluate(test_loader)
        export_confusion_matrix_to_latex(y_test, y_pred, rp.PredictionTask.BINARY_CLASSIFICATION)

        print("\\begin{verbatim}")
        print('***Classification report***')
        print(skl.classification_report(y_test, y_pred))
        print("\\end{verbatim}")

    def evaluate(self, test_loader):
        print('Evaluating binary classification model...')
        self.learner.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test_loader = DeviceDataLoader(test_loader, device)

        y_pred = None
        with torch.no_grad():
            for xb_test, yb_test in test_loader:
                yb_pred = self.learner(xb_test.float())

                yb_pred = yb_pred.detach().cpu().numpy()
                yb_pred = yb_pred.reshape(-1,1)

                yb_test = yb_test.detach().cpu().numpy()
                yb_test = yb_test.reshape(-1,1)

                if y_pred is None:
                    y_pred = yb_pred
                    y_true = yb_test
                else:
                    y_pred = np.vstack([y_pred, yb_pred])
                    y_true = np.vstack([y_true, yb_test])

        y_pred = y_pred.round().ravel()
        assert np.all(np.logical_or(y_pred == 0, y_pred == 1))

        y_true = rp.value_to_binary_level(y_true)
        assert np.all(np.logical_or(y_true == 0, y_true == 1))
        y_true = y_true.ravel()

        return y_true, y_pred