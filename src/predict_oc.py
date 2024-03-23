import torch
import torch.nn as nn
from train.ordinal_classifier import OrdinalClassifier
from train.lstm_neural_net import LstmNeuralNet
import globals as globals
import pickle
import yaml

def predict_oc(pipeline_id: str, prediction_task_sufix: str):

    #
    # Load numpy arrays (stored in a pickle file) from disk
    filename = globals.DATASETS_DIR + pipeline_id + ".pickle"
    file = open(filename, 'rb')
    (_, _, _, _, X_test, _) = pickle.load(file)
    print(f"Shape of test data matrix: {X_test.shape}")

    # Example to predict is the firs one in the test dataset.
    x = X_test[0:1, :, :]
    print(f"Shape of one test example: {x.shape}")
    print("Model input:")
    print(x)

    NUM_FEATURES = X_test.shape[2]

    with open('./config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    SEQ_LENGTH = config["preproc"]["SLIDING_WINDOW_SIZE"]
    DROPOUT_RATE = config["training"][prediction_task_sufix]["DROPOUT_RATE"]
    OUTPUT_SIZE = config["training"][prediction_task_sufix]["OUTPUT_SIZE"]

    #Create an instance of the model
    learner = LstmNeuralNet(seq_length = SEQ_LENGTH,
                        input_size = NUM_FEATURES, 
                        output_size = OUTPUT_SIZE,
                        dropout_rate = DROPOUT_RATE)

    input_dim = (NUM_FEATURES, config["preproc"]["SLIDING_WINDOW_SIZE"])
    model = OrdinalClassifier(learner)

    
    # Load the pretrained model weights from the file
    learner_name = model.learner.__class__.__name__
    model_path = globals.MODELS_DIR + f"best_{pipeline_id}_{prediction_task_sufix}_{learner_name}.pt"
    model.learner.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


    # Set the model in evaluation (inference) mode.
    model.learner.eval()

    # Make prediction using the loaded model
    with torch.no_grad():
        output = model.predict(x)
        print(f"Predicted level: {output[0][0]}")
        return output[0][0]

if __name__ == "__main__":
    pipeline_id = 'A652_A621_A636_A627'
    prediction_task_sufix = "oc"
    # hardcoded for now
    predict_oc(pipeline_id, prediction_task_sufix)