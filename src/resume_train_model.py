import torch
import torch.nn as nn
from train.ordinal_classification_net import OrdinalClassificationNet
import globals as globals
import pickle
import train.pipeline as pipeline

pipeline_id = "A652_N"

X_train, y_train, X_val, y_val, X_test, y_test = pipeline.load_datasets(pipeline_id)

NUM_FEATURES = X_test.shape[2]
NUM_CLASSES = 5

#Create an instance of the model
model = OrdinalClassificationNet(in_channels=NUM_FEATURES, num_classes=NUM_CLASSES)

# Load the pretrained model weights from the file
model_path = globals.MODELS_DIR + "best_" + pipeline_id + "_OC.pt"  # Path to the pretrained model file
model.load_state_dict(torch.load(model_path))


# Make prediction using the loaded model
with torch.no_grad():
    output = model.predict(x)
    print(f"Predicted level: {output[0][0]}")