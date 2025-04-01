import os
import shutil
import cv2
import torch
import argparse
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to compute the focus measure of the image using Laplacian
def variance_of_laplacian(image):  
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Detect blurry images and move them to a "Blurry" folder
def detect_blurry_images(input_folder, threshold=100.0):  # Default threshold is 100.0
    blurry_folder = os.path.join(input_folder, "Blurry")  # The "Blurry" folder in the input directory
    
    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)

    # Loop over the images in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        
        # Skip if it's not an image
        if not (image_name.lower().endswith("jpg") or image_name.lower().endswith("png")):
            continue

        # Read and process the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)

        # If the image is blurry, move it to the "Blurry" folder
        if fm < threshold:
            print(f"{image_name} is blurry.")
            # Move the blurry image to the "Blurry" folder
            destination_image_path = os.path.join(blurry_folder, image_name)
            shutil.move(image_path, destination_image_path)  # Move instead of writing and deleting

# Function to perform model-based classification
def model_based_classification(input_folder, model_path):
    blurry_folder = os.path.join(input_folder, "Blurry")
    
    if not os.path.exists(blurry_folder):
        os.makedirs(blurry_folder)

    blurry_folder_new = os.path.join(blurry_folder, "Blurry")

    if not os.path.exists(blurry_folder_new):
        os.makedirs(blurry_folder_new)

    # Load the trained model
    trained_model = torch.load(model_path)
    trained_model = trained_model['model_state']
    trained_model = trained_model.to(device)
    
    # Loop through the images in the Blurry folder
    img_list = os.listdir(blurry_folder)
    for ind, image_name in enumerate(img_list):
        print(f"Processing image {ind + 1} / {len(img_list)}: {image_name}")
        
        image_path = os.path.join(blurry_folder, image_name)
        
        # Read the image for prediction
        img = cv2.imread(image_path, 0)  # Read as grayscale
        
        prediction = is_image_blurry(trained_model, img, threshold=0.5)
        
        # If the image is classified as blurry by the model, move it to the new "Blurry" folder
        if prediction:
            print(f"Model classified {image_name} as blurry.")
            destination_image_path = os.path.join(blurry_folder_new, image_name)
            
            # Move the image to the new "Blurry" folder
            shutil.move(image_path, destination_image_path)

# Function to check if the image is blurry based on the model
def is_image_blurry(trained_model, img, threshold=0.5):
    feature_extractor = featureExtractor()
    accumulator = []

    # Resize the image by the downsampling factor
    feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])

    # Compute the image ROI using local entropy filter
    feature_extractor.compute_roi()

    # Extract the blur features using DCT transform coefficients
    extracted_features = feature_extractor.extract_feature()
    extracted_features = np.array(extracted_features)

    if len(extracted_features) == 0:
        return True
    
    # Prepare data loader
    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data
        x = x.to(device).float()

        output = trained_model(x)
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())

    # Determine if the image is blurry based on threshold
    prediction = np.mean(accumulator) < threshold
    return prediction

# Main function to process the images
def process_images(args):
    input_folder = args["images"]
    threshold = args["threshold"]
    model_based = args["modelbased"]
    model_path = args["model"]

    # Step 1: Detect blurry images and move them to the "Blurry" folder
    detect_blurry_images(input_folder, threshold)

    # Step 2: If the model-based classification is selected, run it
    if model_based:
        if os.path.exists(os.path.join(input_folder, "Blurry")):
            print("Running model-based classification on blurry images...")
            model_based_classification(input_folder, model_path)

# Parse command-line arguments
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
    ap.add_argument("-t", "--threshold", type=float, default=100.0, help="threshold for blurry image detection")
    ap.add_argument("-m", "--model", required=True, help="path to the pre-trained model")
    ap.add_argument("-mb", "--modelbased", action='store_true', help="run model-based classification on blurry images")
    args = vars(ap.parse_args())

    # Run the process images function
    process_images(args)
