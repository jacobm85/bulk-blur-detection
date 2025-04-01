import os
import argparse
import cv2
from tensorflow.keras.models import load_model

def detect_blurry_images(input_folder, threshold=100.0):  # Default threshold is 100.0
    blurry_folder = os.path.join(input_folder, "Blurry")  # The "Blurry" folder in the input directory

    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)

    # Process all images in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Read the image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Compute the Laplacian of the image to check blurriness
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()

            # If the focus measure is below the threshold, consider the image blurry
            if fm < threshold:
                print(f"{image_name} is blurry.")
                # Move the blurry image to the "Blurry" folder
                cv2.imwrite(os.path.join(blurry_folder, image_name), image)
                os.remove(image_path)  # Remove the original image from the input folder

def model_based_classification(input_folder, model_path):
    blurry_folder = os.path.join(input_folder, "Blurry")  # Get the "Blurry" folder path

    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)

    # Load the pre-trained model
    model = load_model(model_path)

    # Process all images in the "Blurry" folder
    for image_name in os.listdir(blurry_folder):
        image_path = os.path.join(blurry_folder, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))  # Resize for model input
            image = image / 255.0  # Normalize image

            # Add batch dimension and predict
            image = image.reshape((1, 224, 224, 3))
            prediction = model.predict(image)

            # Check prediction and move blurry images accordingly
            if prediction[0][0] > 0.5:  # Assuming binary classification (blurry or not)
                print(f"{image_name} classified as blurry.")
                # Move the blurry image to the same folder
                cv2.imwrite(os.path.join(blurry_folder, image_name), image)
                os.remove(image_path)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process blurry images")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing images")
    parser.add_argument("-t", "--threshold", type=float, default=100.0, help="Threshold for Laplacian blurriness detection")
    parser.add_argument("-m", "--model", help="Path to the pre-trained model")
    parser.add_argument("-f", "--final_folder", help="Folder to store final blurry images", required=False)
    
    args = parser.parse_args()

    # Detect blurry images
    detect_blurry_images(args.input, args.threshold)

    # If the model path is provided, run model-based classification
    if args.model:
        model_based_classification(args.input, args.model)

    # If a final folder is provided, move the images to the final folder
    if args.final_folder:
        # Ensure the final folder exists
        if not os.path.exists(args.final_folder):
            os.mkdir(args.final_folder)

        # Move blurry images from the "Blurry" folder to the final folder
        blurry_folder = os.path.join(args.input, "Blurry")
        for image_name in os.listdir(blurry_folder):
            blurry_image_path = os.path.join(blurry_folder, image_name)
            if os.path.isfile(blurry_image_path):
                final_image_path = os.path.join(args.final_folder, image_name)
                os.rename(blurry_image_path, final_image_path)

if __name__ == "__main__":
    main()
