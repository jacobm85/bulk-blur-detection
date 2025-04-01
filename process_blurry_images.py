import os
import cv2
import numpy as np
import argparse
from keras.models import load_model

def detect_blurry_images(input_folder, threshold=100.0):
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
    
    model = load_model(model_path)
    
    for image_name in os.listdir(blurry_folder):
        image_path = os.path.join(blurry_folder, image_name)
        
        if not os.path.isfile(image_path):
            continue
        
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (224, 224))  # Resize image for model input
        image_normalized = np.expand_dims(image_resized, axis=0) / 255.0
        
        # Predict the class of the image
        prediction = model.predict(image_normalized)
        
        if prediction[0][0] > 0.5:  # Assuming 0.5 threshold for binary classification
            print(f"{image_name} classified as blurry by model.")
            os.rename(image_path, os.path.join(blurry_folder_new, image_name))  # Move the image

def process_images(input_folder, threshold, model_path=None, modelbased=False):
    detect_blurry_images(input_folder, threshold)
    
    if modelbased and model_path:
        model_based_classification(input_folder, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', required=True, help="Input folder with images")
    parser.add_argument('-t', '--threshold', default=19.0, type=float, help="Threshold for blur detection")
    parser.add_argument('-m', '--model', help="Path to the trained model for model-based classification")
    parser.add_argument('-mb', '--modelbased', action='store_true', help="Enable model-based classification")
    
    args = parser.parse_args()
    
    process_images(args.images, args.threshold, args.model, args.modelbased)
