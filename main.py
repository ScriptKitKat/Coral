from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import io
import numpy as np

app = Flask(__name__)

# The neural network to train the model
class CoralCNN(nn.Module):
  def __init__(self):
    super(CoralCNN, self).__init__()
    self.conv_stack = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(256),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fc1 = nn.Linear(1024, 512)
    self.dropout = nn.Dropout(p=0.5)
    self.fc2 = nn.Linear(512, 2)
    self.flatten = nn.Flatten()

  def forward(self, x):
    x = self.conv_stack(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = nn.ReLU()(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x

def load_checkpoint(model, optimizer, latest_checkpoint_path):
    if os.path.isfile(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_train_loss = checkpoint['train_loss']
        print(f'Checkpoint loaded, resuming from epoch {start_epoch}')
        return model, optimizer, start_epoch, best_train_loss
    else:
        print('No checkpoint found, starting from scratch')
        return model, optimizer, 0, float('inf')

model = CoralCNN()
#
# optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-5)
#
# model, optimizer, start_epoch, best_train_loss = load_checkpoint(model, optimizer, 'coral_cnn_model.pth')
#
# model.load_state_dict(torch.load('coral_cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            img_bytes = file.read()
            img_tensor = transform_image(img_bytes)
            prediction = get_prediction(img_tensor)
            return jsonify({'class_id': prediction})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
