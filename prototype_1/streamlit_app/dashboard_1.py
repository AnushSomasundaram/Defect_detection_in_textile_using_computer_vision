import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

# Function 1: Compare contours and create a canvas
def compare_contours_canvas_creator(model, test_piece, file_path):
    # Read the first image
    image1 = cv2.imread(model)

    # Read the second image
    image2 = cv2.imread(test_piece)

    # Resize the images to the same size
    resized_image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    resized_image1 = image1

    # Convert images to grayscale
    gray1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Perform edge detection using Canny
    edges1 = cv2.Canny(blurred1, 50, 150)
    edges2 = cv2.Canny(blurred2, 50, 150)

    # Find contours in the first edge image
    contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours in the second edge image
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank canvas with the size of the larger image
    canvas_height = max(resized_image1.shape[0], resized_image2.shape[0])
    canvas_width = max(resized_image1.shape[1], resized_image2.shape[1])
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Draw the contours from the first image onto the canvas using green color
    cv2.drawContours(canvas, contours1, -1, (0, 255, 0), 2)

    # Draw the contours from the second image onto the canvas using red color
    cv2.drawContours(canvas, contours2, -1, (0, 0, 255), 2)

    # Save the canvas as an image file
    cv2.imwrite(file_path, canvas)

# Function 2: Predict using VGG16 model
def predict_vgg16_single(image_path, model_path):
    # Load custom VGG model trained on Azhar's PC
    model = models.vgg16(pretrained=False)
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Preprocess the image
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

    # Get the predicted class index
    _, predicted_idx = torch.max(outputs, 1)
    predicted_index = predicted_idx.item()

    return predicted_index

# Main function
def main():
    st.header("Image Processing and Prediction")

    # Upload an image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save the uploaded image
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Display the uploaded image
        st.subheader("Uploaded Image")
        st.image(uploaded_image)

        
        # Process the image using Function 1
        canvas_path = "canvas_image.jpg"
        compare_contours_canvas_creator("/Users/software/Desktop/ai-in-textile/images/perfect_sample.png", image_path, canvas_path)

        # Display the processed image
        st.subheader("Processed Image")
        processed_image = Image.open(canvas_path)
        st.image(processed_image)

        # Run Function 2 on the processed image
        st.subheader("Prediction")
        predicted_index = predict_vgg16_single(image_path, "/Users/software/Desktop/ai-in-textile/prototype_1/pytorch_model/shape1_vgg16_model.pth")
        if predicted_index == 0:
            st.success("Correct!")
        else:
            st.error("Incorrect!")

        # Provide a download link for the processed image
        st.download_button("Download Image", canvas_path)

if __name__ == "__main__":
    main()
