import os
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# Constants
MODEL_PATH = "resnet_classifier.pth"
NUM_CLASSES = 5
LABELS = ["Ashkentoz", "Koshi", "ching-chong", "pocahontas", "Terrorist/Mizrahol"]

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load the trained model
class ResNetClassifier(torch.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Load the model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def predict(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(image)

    # Get probabilities using softmax
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)

    # Get the predicted class and confidence
    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100  # Convert to percentage

    return LABELS[predicted_class], confidence


def main():
    # Ask for the photo location
    image_path = input("Enter the image file path: ")

    # Check if the file exists
    if not os.path.isfile(image_path):
        print("File not found!")
        return

    # Get the prediction
    label, confidence = predict(image_path)
    print(f"The image is most likely {label} with {confidence:.2f}% confidence.")


if __name__ == "__main__":
    main()
