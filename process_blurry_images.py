import os
import torch
import cv2
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser to handle input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0, help="focus measures that fall below this value will be considered 'blurry'")
ap.add_argument("-m", "--model", required=True, help="path to the trained model")
ap.add_argument("-mb", "--modelbased", type=bool, default=False, help="whether to use model-based classification")
args = vars(ap.parse_args())

# Load the trained model
trained_model = torch.load(args['model'])
trained_model = trained_model['model_state']

def detect_blurry_images(input_folder, threshold=100.0):  # Default threshold is 100.0
    blurry_folder = os.path.join(input_folder, "Blurry")  # The "Blurry" folder in the input directory
    
    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)

    # Loop through the images in the "Blurry" folder to detect blurry images
    for image_name in os.listdir(input_folder):
        if not (image_name.lower().endswith("jpg") or image_name.lower().endswith("png")):
            continue

        # Load the image from the input folder
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)

        if fm < threshold:
            print(f"{image_name} is blurry.")
            # Move the blurry image to the "Blurry" folder
            cv2.imwrite(os.path.join(blurry_folder, image_name), image)
            os.remove(image_path)  # Remove the original image from the input folder

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def run_testing_on_dataset(trained_model, dataset_dir, GT_blurry):
    correct_prediction_count = 0
    img_list = os.listdir(dataset_dir)
    for ind, image_name in enumerate(img_list):
        print(f"Blurry Image Prediction: {ind+1} / {len(img_list)} images processed..")

        # Read the image
        img = cv2.imread(os.path.join(dataset_dir, image_name), 0)

        prediction = is_image_blurry(trained_model, img, threshold=0.5)

        if prediction == GT_blurry:
            correct_prediction_count += 1
    accuracy = correct_prediction_count / len(img_list)
    return accuracy

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
    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data
        x = x.to(device).float()

        output = trained_model(x)
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())

    prediction = np.mean(accumulator) < threshold
    return prediction

def model_based_classification(input_folder, model_path):
    # Define the path for the first "Blurry" folder (inside the input folder)
    blurry_folder = os.path.join(input_folder, "Blurry")
    
    # Check if the first-level "Blurry" folder exists
    if not os.path.exists(blurry_folder):
        print("No blurry images found. Skipping model-based classification.")
        return

    # Run the model-based classification on the first-level "Blurry" folder
    accuracy_blurry_images = run_testing_on_dataset(trained_model, blurry_folder, GT_blurry=True)
    print(f"Test accuracy on blurry folder = {accuracy_blurry_images}")

if __name__ == '__main__':
    # First, classify blurry images using basic threshold
    detect_blurry_images(args["images"], threshold=args["threshold"])

    # If the user checked the checkbox to do model-based classification, run it
    if args["modelbased"]:
        model_based_classification(args["images"], model_path=args["model"])
