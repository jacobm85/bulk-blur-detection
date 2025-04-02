import os
import torch
import cv2
import shutil
import numpy as np
import argparse
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import Dataset, DataLoader
from torch.serialization import add_safe_globals
from torch.nn import Linear  # Example of custom layer (update based on your model's custom layers)


# Add the MLP class to the safe globals (make sure you import MLP if it's not already imported)
from utils.MLP import MLP

# Add safe globals before loading the model
torch.serialization.add_safe_globals([MLP])

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

def run_testing_on_dataset(trained_model, dataset_dir, model_threshold=0.5, GT_blurry=True):
    new_input_folder = os.path.join(dataset_dir, "blurry")
    blurry_folder = os.path.join(new_input_folder, "Blurry")
    sharp_folder = os.path.join(new_input_folder, "Sharp")
    
    # Create directories if they don't exist
    os.makedirs(blurry_folder, exist_ok=True)
#    os.makedirs(sharp_folder, exist_ok=True)
    
    correct_prediction_count = 0
    img_list = os.listdir(new_input_folder)
    for ind, image_name in enumerate(img_list):
        print("Blurry Image Prediction: %d / %d images processed.." % (ind, len(img_list)))
        image_path = os.path.join(new_input_folder, image_name)
        # Read the image
        img = cv2.imread(image_path, 0)
        
        # Validate the source folder and threshold
        if not os.path.exists(image_path):
            return f"File does not exist: {image_path}!", 400
        
        if img is None:
            print(f"Error reading image: {image_path}")
            continue
        
        if not os.path.exists(image_path):
            print(f"File does not exist: {image_path}")
            continue
            
        prediction = is_image_blurry(trained_model, img, model_threshold)

        # Move the image to the appropriate folder based on the prediction
        if prediction:  # If the image is predicted as blurry
            print(f"Yes, {image_name} is blurry.")
            shutil.move(os.path.join(new_input_folder, image_name), os.path.join(blurry_folder, image_name))
        else:  # If the image is predicted as sharp
            print(f"{image_name} is sharp.")
            #shutil.move(os.path.join(new_input_folder, image_name), os.path.join(sharp_folder, image_name))
    
    accuracy = correct_prediction_count / len(img_list)
    return accuracy

def is_image_blurry(trained_model, img, model_threshold):
    feature_extractor = featureExtractor()
    accumulator = []

    # Print the shape of the image for debugging
    if img is None:
        print("Error: Image is None")
        return True

    print(f"Image shape: {np.shape(img)}")
    
    # Resize the image by the downsampling factor
    if img is not None and len(np.shape(img)) == 2:
        feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])
    else:
        print("Error: Image does not have expected dimensions")
        return True    
   
    #feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])

    # compute the image ROI using local entropy filter
    feature_extractor.compute_roi()

    # extract the blur features using DCT transform coefficients
    extracted_features = feature_extractor.extract_feature()
    extracted_features = np.array(extracted_features)

    if(len(extracted_features) == 0):
        return True
    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    # trained_model.test()
    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data
        x = x.to(device).float()

        output = trained_model(x)
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())

    prediction = np.mean(accumulator) < model_threshold
    return prediction

def process_images(input_folder, threshold, model_threshold, model_path=None, modelbased=False):
    # Step 1: Detect blurry images based on Laplacian variance
    detect_blurry_images(input_folder, threshold)
    
    # Step 2: If model-based classification is enabled, classify using the PyTorch model
    if modelbased:
        # Add safe globals for custom classes
        add_safe_globals([Linear])  # Replace Linear with any custom class used in your model
        
        trained_model = torch.load(model_path, weights_only=False)
        trained_model = trained_model['model_state'] if isinstance(trained_model, dict) else trained_model
        
        run_testing_on_dataset(trained_model, input_folder, model_threshold, GT_blurry=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', required=True, help="Input folder with images")
    parser.add_argument('-t', '--threshold', default=19.0, type=float, help="Threshold for blur detection")
    parser.add_argument('-mt', '--model_threshold', default=0.5, type=float, help="Threshold for model-based classification")
    parser.add_argument('-m', '--model', help="Path to the trained PyTorch model for model-based classification")
    parser.add_argument('-mb', '--modelbased', action='store_true', help="Enable model-based classification")
    
    args = parser.parse_args()
    
    process_images(args.images, args.threshold, args.model_threshold, args.model, args.modelbased)
