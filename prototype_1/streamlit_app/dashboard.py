import streamlit as st
from PIL import Image

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



# # Function to process the uploaded image
# def process_image(image):
#     # Add your image processing logic here
#     # Return True if correct, False otherwise

def convert_classification_into_label(image_path,model_path):
    predicted_class =int( predict_vgg16_single(image_path,model_path))
    if predicted_class==0:
        return True
    else:
        return False

# Set up the Streamlit app
def main():
    # Create a navigation bar
    st.sidebar.title("Navigation")
    pages = ["Upload Image"]
    choice = st.sidebar.selectbox("Go to", pages)

    if choice == "Upload Image":
        st.title("Image Processing Dashboard")
        st.write("Upload an image and click the 'Process' button.")

        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button('Process'):
                is_correct = convert_classification_into_label(uploaded_file,model_path = "/Users/software/Desktop/ai-in-textile/prototype_1/pytorch_model/shape1_vgg16_model.pth")

            if is_correct:
                st.success("✅ Correct!")
            else:
                st.error("❌ Incorrect!")
if __name__ == '__main__':
    main()