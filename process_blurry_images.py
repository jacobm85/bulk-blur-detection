import cv2
import os
import torch
import numpy as np
import shutil
import argparse
from torch.utils.data import Dataset, DataLoader
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset

# Define device for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to compute variance of Laplacian (focus measure)
def variance_of_laplacian(image):  
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Function for initial blurriness detection using Laplacian variance
def detect_blurry_images(input_folder, threshold=100.0):  # Default threshold is 100.0
    blurry_folder = os.path.join(input_folder, "Blurry")  # The "Blurry" folder in the input directory
    
    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)
    
    # Loop through the images in the folder
    for image_name in os.listdir(input_folder):
        if not (image_name.lower().endswith("jpg") or image_name.lower().endswith("png")):
            continue

        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)

        # Debugging: print the focus measure (Laplacian variance) for each image
        print(f"Focus measure for {image_name}: {fm}")

        # If the focus measure is below the threshold, move to blurry folder
        if fm < threshold:
            print(f"{image_name} is blurry.")
            cv2.imwrite(os.path.join(blurry_folder, image_name), image)
            os.remove(image_path)  # Remove from the original folder
        else:
            print(f"{image_name} is not blurry (focus measure: {fm}).")

# Function to check if an image is blurry using the pre-trained model
def is_image_blurry(trained_model, img, threshold=0.5):
    feature_extractor = featureExtractor()
    accumulator = []

    # Resize and compute the image ROI
    feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])
    feature_extractor.compute_roi()

    # Extract features using DCT transform coefficients
    extracted_features = feature_extractor.extract_feature()
    extracted_features = np.array(extracted_features)

    if len(extracted_features) == 0:
        return True
    
    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data.to(device).float()

        output = trained_model(x)
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())

    prediction = np.mean(accumulator) < threshold
    return prediction

# Function to run the testing on a dataset (recheck blurry images using the model)
def recheck_blurry_images(blurry_folder, trained_model):
    final_blurry_folder = os.path.join(blurry_folder, "Blurry")  # "Blurry" folder within the blurry_folder

    if not os.path.exists(final_blurry_folder):
        os.mkdir(final_blurry_folder)

    # Process each image in the blurry folder
    for image_name in os.listdir(blurry_folder):
        image_path = os.path.join(blurry_folder, image_name)
        
        # Skip non-image files
        if not (image_name.lower().endswith("jpg") or image_name.lower().endswith("png")):
            continue
        
        # Read the image in grayscale
        img = cv2.imread(image_path, 0)
        
        # Use the model to check if the image is blurry
        is_blurry = is_image_blurry(trained_model, img, threshold=0.5)

        if is_blurry:
            print(f"{image_name} is confirmed blurry. Moving to {final_blurry_folder}")
            # Confirmed blurry, move it to the final blurry folder (within blurry_folder)
            shutil.move(image_path, os.path.join(final_blurry_folder, image_name))
        else:
            print(f"{image_name} is not blurry. Removing from {blurry_folder}")
            # If not blurry, remove it from the "Blurry" folder
            os.remove(image_path)

# Main function to execute both stages (initial detection + reclassification)
def process_images(input_folder, model_path, laplacian_threshold=100.0):
    # Load the pre-trained model
    trained_model = torch.load(model_path)
    trained_model = trained_model['model_state']
    trained_model.eval()

    # Step 1: Detect blurry images using Laplacian
    detect_blurry_images(input_folder, threshold=laplacian_threshold)

    # Step 2: Recheck the blurry images using the pre-trained model
    blurry_folder = os.path.join(input_folder, "Blurry")  # "Blurry" folder located in input_folder
    recheck_blurry_images(blurry_folder, trained_model)

# Parse arguments using argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images for blurriness detection and classification.")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the input folder containing images.")
    parser.add_argument("-t", "--threshold", type=float, default=100.0, help="Threshold for Laplacian focus measure (default 100.0).")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the pre-trained model.")
    
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Run the image processing with the parsed arguments
    process_images(
        input_folder=args.input_folder,
        model_path=args.model_path,
        laplacian_threshold=args.threshold
    )
