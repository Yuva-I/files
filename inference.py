import torch
import numpy as np
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import ast  # Library for parsing string literals to Python data structures
import torch
import numpy as np
from torchvision import transforms

class MalConv(nn.Module):
    def __init__(self):
        super(MalConv, self).__init__()
        self.fc1 = nn.Linear(2381, 1000)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1000, 1)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = torch.sigmoid(x)
        return x


# Define any preprocessing transformations if needed
"""preprocess = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations as needed
])"""

# Load the PyTorch model
def load_model(model_path):
    model = MalConv()  # Instantiate your PyTorch model class
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

# Perform inference on input data
def inference(model, input_data):
    # Preprocess input data
    input_data = torch.tensor(input_data)
    # Perform inference
    with torch.no_grad():
        output = model(input_data)
    # Post-process output if needed
    #print(output)
    return output.numpy()  # Convert output tensor to numpy array

# Example function to handle incoming requests
def predict(input_data):
    # Load the model
    model = load_model('model.pt')  # Path to the trained model
    # Perform inference
    result = inference(model, input_data)
    return result
def main():
    """if len(sys.argv) != 2:
        print("Usage: python test_inference.py \"[[1, 2, 3, ..., 2381]]\"")
        sys.exit(1)"""

    input_data_str = sys.argv[1]
    try:
        input_data = ast.literal_eval(input_data_str)
    except ValueError:
        print("Error: Invalid input format. Input must be a valid 2D list.")
        sys.exit(1)

    # Ensure that the input data shape is [1, 2381]
    if not isinstance(input_data, list) or len(input_data) != 1 or not isinstance(input_data[0], list) or len(input_data[0]) != 2381:
        print("Error: Input data shape must be [1, 2381].")
        sys.exit(1)
    #print(input_data)
    result = predict(input_data)
    print("Inference result:", result)
    return result

if __name__ == "__main__":
    main()