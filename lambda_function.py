import json
import numpy as np
from minio import Minio
import os
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn



target_size = (64, 64)
# Define the CNN model
class CustomCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Calculate the flattened size
        self.fc1_input_size = self.calculate_fc1_input_size((3, 64, 64))  # Adjust input size as needed

        self.fc1 = nn.Linear(self.fc1_input_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def calculate_fc1_input_size(self, input_size):
        # Calculate the size of the input to fc1
        with torch.no_grad():
            x = torch.randn(1, *input_size)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = self.pool(torch.relu(self.conv5(x)))
            x = self.pool(x)
            x = x.view(1, -1)
        return x.size(1)

# Initialize MinIO client
minio_client = Minio(
    os.environ['MINIO_ENDPOINT'],
    access_key=os.environ['MINIO_ACCESS_KEY'],
    secret_key=os.environ['MINIO_SECRET_KEY'],
    secure=False  # Set to True if using HTTPS
)

print("minio client initialized:")
# Load the model from MinIO
def load_model():
    response = minio_client.get_object('birdwatcher-models', 'second_model.pth')
    model_data = response.read()
    
    # model = torch.load('E:/HAJNI/birdwatcher/project_classification/models/chicken_or_duck/model_all_v2.pth', map_location=torch.device('cpu'))
    model = CustomCNNModel(2)
    state_dict = torch.load(io.BytesIO(model_data), map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

def lambda_handler(event, context):
    bucket = event['bucket']
    image_key = event['image_key']
    try:
        # Get the image from MinIO
        response = minio_client.get_object(bucket, image_key)
        image_data = response.read()
        image = Image.open(io.BytesIO(image_data))


        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)
            _, predicted = torch.max(output, 1)
            class_id = predicted.item()
        
        # Map class ID to class names
        class_names = ['chicken', 'duck']
        class_name = class_names[class_id]
        # Process the output and return results
        print(f"Class ID: {class_id}, Class Name: {class_name}")
        return {
            'statusCode': 200,
            'body': json.dumps(class_name)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps('Internal Server Error')
        }
