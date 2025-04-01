import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model

# Function for detecting blurry images
def detect_blurry_images(input_folder, threshold=100.0):
    blurry_folder = os.path.join(input_folder, "Blurry")
    
    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)
    
    # Loop through images in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)

        if os.path.isdir(image_path):
            continue  # Skip subdirectories

        image = cv2.imread(image_path)
        fm = cv2.Laplacian(image, cv2.CV_64F).var()  # Focus measure

        if fm < threshold:
            print(f"{image_name} is blurry.")
            # Move the blurry image to the "Blurry" folder
            os.rename(image_path, os.path.join(blurry_folder, image_name))  # Move instead of rewriting

# Function for model-based classification
def model_based_classification(input_folder, model_path):
    blurry_folder = os.path.join(input_folder, "Blurry")
    blurry_folder_new = os.path.join(blurry_folder, "Blurry")  # New folder for model classification

    if not os.path.exists(blurry_folder_new):
        os.mkdir(blurry_folder_new)
    
    model = load_model(model_path)
    
    # Classify images in the blurry folder
    for image_name in os.listdir(blurry_folder):
        image_path = os.path.join(blurry_folder, image_name)

        if os.path.isdir(image_path):
            continue  # Skip subdirectories

        # Image preprocessing and prediction steps here
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Assuming model expects 224x224 input size
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        
        if prediction[0][0] > 0.5:  # Assuming binary classification
            os.rename(image_path, os.path.join(blurry_folder_new, image_name))

def main():
    parser = argparse.ArgumentParser(description="Process blurry images.")
    parser.add_argument('-i', '--input', required=True, help="Input folder containing images")
    parser.add_argument('-t', '--threshold', type=float, default=100.0, help="Threshold for detecting blurry images")
    parser.add_argument('-m', '--model', required=False, help="Path to the pre-trained model for classification")
    
    args = parser.parse_args()
    
    detect_blurry_images(args.input, threshold=args.threshold)

    # If model path is provided, run model-based classification
    if args.model:
        model_based_classification(args.input, model_path=args.model)

if __name__ == '__main__':
    main()
