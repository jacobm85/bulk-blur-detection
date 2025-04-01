import os
import cv2
import numpy as np
import argparse
import torch
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detect_blurry_images(input_folder, threshold=19.0):
    blurry_folder = os.path.join(input_folder, "Blurry")
    
    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)
    
    # Loop through all images in the folder and check blurriness
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        
        if not os.path.isfile(image_path):
            continue

        # Read the image
        image = cv2.imread(image_path)
        
        # Calculate the Laplacian variance to check for blurriness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if fm < threshold:
            print(f"{image_name} is blurry.")
            # Move the blurry image to the "Blurry" folder
            os.rename(image_path, os.path.join(blurry_folder, image_name))  # Move the image

def model_based_classification(input_folder, model_path):
    blurry_folder = os.path.join(input_folder, "Blurry")
    blurry_folder_new = os.path.join(blurry_folder, "Blurry")

    if not os.path.exists(blurry_folder_new):
        os.mkdir(blurry_folder_new)
    
    # Load the trained PyTorch model
    trained_model = torch.load(model_path)
    trained_model = trained_model['model_state']
    trained_model.eval()  # Set the model to evaluation mode
    
    # Loop through all images in the "Blurry" folder
    for image_name in os.listdir(blurry_folder):
        image_path = os.path.join(blurry_folder, image_name)
        
        if not os.path.isfile(image_path):
            continue
        
        # Read and process the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(image, (224, 224))  # Resize image for model input
        image_normalized = image_resized / 255.0
        image_normalized = torch.tensor(image_normalized).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        
        # Make prediction
        with torch.no_grad():
            output = trained_model(image_normalized.to(device))
        
        # Assuming binary classification with output [0] being blurry and [1] being sharp
        prediction = torch.sigmoid(output).item()  # Assuming binary classification
        
        if prediction > 0.5:  # If blurry image as per model's prediction
            print(f"{image_name} classified as blurry by model.")
            os.rename(image_path, os.path.join(blurry_folder_new, image_name))  # Move the image

def process_images(input_folder, threshold, model_path=None, modelbased=False):
    # Step 1: Detect blurry images based on Laplacian variance
    detect_blurry_images(input_folder, threshold)
    
    # Step 2: If model-based classification is enabled, classify using the PyTorch model
    if modelbased and model_path:
        model_based_classification(input_folder, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', required=True, help="Input folder with images")
    parser.add_argument('-t', '--threshold', default=19.0, type=float, help="Threshold for blur detection")
    parser.add_argument('-m', '--model', help="Path to the trained PyTorch model for model-based classification")
    parser.add_argument('-mb', '--modelbased', action='store_true', help="Enable model-based classification")
    
    args = parser.parse_args()
    
    process_images(args.images, args.threshold, args.model, args.modelbased)
