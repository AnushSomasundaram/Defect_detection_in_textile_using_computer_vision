import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Define the path to your dataset folder
data_path = "/Users/software/Desktop/ai-in-textile/prototype_1/contour_contrast_images/shape1/augemented_images"

# Define the transformation to apply to the images (e.g., resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize if needed
])

# Create an instance of ImageFolder and apply the transformation
dataset = (ImageFolder(root=data_path, transform=transform))

# Create a data loader to load the images in batches during training or evaluation
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate over the data loader to access the images and labels
# for images, labels in data_loader:
#     # Process the images and labels as needed for your task
#     # e.g., pass them through your model for training or evaluation
#     pass

model = vgg16(pretrained=True)

num_classes = len(dataset.classes)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)

device = torch.device("mps")
if device.type == "mps":
    model = model.to(device)
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

from tqdm import tqdm
num_epochs = 10

for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    i=1
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        print(i)
        i=i+1
    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
torch.save(model.state_dict(),'hieroglyph_vgg_16.pth')



