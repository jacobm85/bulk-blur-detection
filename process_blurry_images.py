import os
import torch
import cv2
import numpy as np
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import DataLoader
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_testing_on_dataset(trained_model, dataset_dir, GT_blurry):
    correct_prediction_count = 0
    img_list = os.listdir(dataset_dir)
    for ind, image_name in enumerate(img_list):
        print("Blurry Image Prediction: %d / %d images processed.." % (ind, len(img_list)))

        # Read the image
        img = cv2.imread(os.path.join(dataset_dir, image_name), 0)

        prediction = is_image_blurry(trained_model, img, threshold=0.5)

        if(prediction == GT_blurry):
            correct_prediction_count += 1
    accuracy = correct_prediction_count / len(img_list)
    return accuracy


def is_image_blurry(trained_model, img, threshold=0.5):
    feature_extractor = featureExtractor()
    accumulator = []

    # Resize the image by the downsampling factor
    feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])

    # compute the image ROI using local entropy filter
    feature_extractor.compute_roi()

    # extract the blur features using DCT transform coefficients
    extracted_features = feature_extractor.extract_feature()
    extracted_features = np.array(extracted_features)

    if len(extracted_features) == 0:
        return True
    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    # Run the model on the input data
    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data
        x = x.to(device).float()

        output = trained_model(x)
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())

    prediction = np.mean(accumulator) < threshold
    return prediction


def model_based_classification(input_folder, model_path):
    blurry_folder = os.path.join(input_folder, "Blurry")

    # Load the trained PyTorch model
    trained_model = torch.load(model_path)
    trained_model = trained_model['model_state']
    trained_model.eval()

    # Perform model-based classification on blurry images
    for image_name in os.listdir(blurry_folder):
        image_path = os.path.join(blurry_folder, image_name)
        img = cv2.imread(image_path, 0)  # Assuming the image is grayscale

        prediction = is_image_blurry(trained_model, img)

        if prediction:  # If the image is classified as blurry, move it
            print(f"{image_name} is blurry!")
            # Here you can move the image, or any other processing you want
            new_location = os.path.join(input_folder, "Blurry", "Processed", image_name)
            os.rename(image_path, new_location)


def main():
    parser = argparse.ArgumentParser(description='Process blurry images and run model-based classification.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input folder path')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model file path')
    parser.add_argument('-t', '--threshold', type=float, default=19.0, help='Threshold for blur classification')
    parser.add_argument('-f', '--final_blurry_folder', type=str, help='Final folder for blurry images')

    args = parser.parse_args()

    # Run the model-based classification if the checkbox is selected (modelbased == True)
    if args.final_blurry_folder:  # Check if final_blurry_folder exists (model-based checkbox is selected)
        model_based_classification(args.input, model_path=args.model)

    # Otherwise, process using the threshold and blur detection
    # This part involves processing images based on the threshold if model-based is not selected

if __name__ == '__main__':
    main()
