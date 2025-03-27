import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# Constants
DATA_DIR = r'F:\Desktop\pythonProject\DataBase'
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
NUM_CLASSES = 5

# Supported image extensions
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp'}


# Function to get valid image files from the directory
def get_valid_image_files(dataset_dir):
    image_files = []

    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                image_files.append(os.path.join(subdir, file))

    return image_files


# Define data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the images to 128x128
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to 3 channels (RGB)
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])


# Load dataset
def load_images(dataset_dir):
    image_files = get_valid_image_files(dataset_dir)
    images = []
    labels = []

    class_labels = {class_name: idx for idx, class_name in enumerate(os.listdir(dataset_dir))}

    for image_path in image_files:
        # Open and transform the image
        img = Image.open(image_path)
        img = transform(img)
        images.append(img)

        class_name = os.path.basename(os.path.dirname(image_path))
        labels.append(class_labels[class_name])

    return torch.stack(images), torch.tensor(labels)


# Get the valid images and labels
images, labels = load_images(DATA_DIR)
dataset = torch.utils.data.TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Define the ResNet-50 model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {running_loss / len(dataloader):.4f}")

# Save trained model
torch.save(model.state_dict(), "resnet_classifier.pth")
