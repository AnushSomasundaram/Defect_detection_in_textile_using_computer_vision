import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16

def predict_vgg16_single(image_path,model_path):
    # load custom vgg model trained on azhars pc
    
    model = models.vgg16(pretrained=False)
    
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
    #preprocessing the image
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_image = Image.open(image_path).convert('RGB')
    
    input_data = preprocess(input_image)
    input_data = input_data.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_data)
        
    #Get the predicted class index
    
    _ , predicted_idx = torch.max(outputs,1)
    predicted_index = predicted_idx.item()
    
    
    return predicted_index


model_path = "/Users/software/Desktop/ai-in-textile/prototype_1/pytorch_model/shape1_vgg16_model.pth"
image_path = "/Users/software/Desktop/ai-in-textile/prototype_1/contour_contrast_images/shape1/defect_pieces/Screenshot 2023-07-14 at 10.49.24 PM.png"
predicted_class= predict_vgg16_single(image_path,model_path)

if predicted_class==0:
    print( "Proceed")
else:
    print( "Discard")


def convert_classification_into_label(image_path,model_path):
    predicted_class =int( predict_vgg16_single(image_path,model_path))
    if predicted_class==0:
        return "Proceed"
    else:
        return "Discard"

#convert_classification_into_label(model_path,image_path)

