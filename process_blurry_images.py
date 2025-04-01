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
    
    # Ensure the "Blurry" folder exists
    if not os.path.exists(blurry_folder):
        print(f"Creating blurry folder at {blurry_folder}")  # Debugging line
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
    sharp_folder = os.path.join(input_folder, "Sharp")

    # Ensure both "Blurry" and "Sharp" folders exist
    if not os.path.exists(blurry_folder):
        print(f"Creating blurry folder at {blurry_folder}")  # Debugging line
        os.mkdir(blurry_folder)

    if not os.path.exists(sharp_folder):
        print(f"Creating sharp folder at {sharp_folder}")  # Debugging line
        os.mkdir(sharp_folder)

    # Try loading the model and handle errors
    try:
        trained_model = torch.load(model_path)
        trained_model = trained_model['model_state']
        trained_model.eval()  # Set the model to evaluation mode
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return
    except KeyError:
        print(f"Error: Model file is missing 'model_state' key.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return
    
    # Loop through all images in the "Blurry" folder
    for image_name in os.listdir(blurry_folder):
        image_path = os.path.join(blurry_folder, image_name)
        
        if not os.path.isfile(image_path):
            continue
        
        try:
            # Read and process the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read image {image_name}")
                continue
            
            image_resized = cv2.resize(image, (224, 224))  # Resize image for model input
            image_normalized = image_resized / 255.0
            image_normalized = torch.tensor(image_normalized).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions

            # Make prediction
            with torch.no_grad():
                output = trained_model(image_normalized.to(device))

            # Assuming binary classification with output [0] being blurry and [1] being sharp
            prediction = torch.sigmoid(output).item()  # Assuming binary classification

            # Move sharp images to the "Sharp" folder and blurry images to the "Blurry" folder
            if prediction > 0.5:  # If blurry image as per model's prediction
                print(f"{image_name} classified as blurry by model.")
                os.rename(image_path, os.path.join(blurry_folder, image_name))  # Move to "Blurry"
            else:  # If sharp image as per model's prediction
                print(f"{image_name} classified as sharp by model.")
                os.rename(image_path, os.path.join(sharp_folder, image_name))  # Move to "Sharp"
        except Exception as e:
            print(f"An error occurred while processing {image_name}: {e}")

def process_images(input_folder, threshold, model_path=None, modelbased=True):
    # Step 1: Detect blurry images based on Laplacian variance
    detect_blurry_images(input_folder, threshold)
    
    # Step 2: If model-based classification is enabled, classify using the PyTorch model
    if modelbased:
        model_based_classification(input_folder, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', required=True, help="Input folder with images")
    parser.add_argument('-t', '--threshold', default=19.0, type=float, help="Threshold for blur detection")
    parser.add_argument('-m', '--model', help="Path to the trained PyTorch model for model-based classification")
    parser.add_argument('-mb', '--modelbased')#, action='store_true', help="Enable model-based classification")
    
    args = parser.parse_args()
    
    process_images(args.images, args.threshold, args.model, args.modelbased)
