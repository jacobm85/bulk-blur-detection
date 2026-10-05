import os
import shutil
import argparse

import cv2
import numpy as np
import torch

from utils.feature_extractor import featureExtractor
from utils.MLP import MLP  # noqa: F401 -- needed so torch.load can unpickle the saved model

BLURRY_FOLDER_NAME = "Blurry"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def safe_move(src, dst_folder):
    """Move src into dst_folder without overwriting an existing file with the same name."""
    name, ext = os.path.splitext(os.path.basename(src))
    dst = os.path.join(dst_folder, name + ext)
    counter = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_folder, f"{name}_{counter}{ext}")
        counter += 1
    shutil.move(src, dst)
    return dst


def list_files(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, f))]


def downscale(gray, max_size):
    """Shrink the image so its longest side is at most max_size (0 = keep original size)."""
    if max_size <= 0:
        return gray
    rows, cols = gray.shape[:2]
    scale = max_size / max(rows, cols)
    if scale >= 1:
        return gray
    return cv2.resize(gray, (int(cols * scale), int(rows * scale)), interpolation=cv2.INTER_AREA)


def detect_blurry_images(input_folder, threshold=19.0, max_size=0):
    blurry_folder = os.path.join(input_folder, BLURRY_FOLDER_NAME)
    os.makedirs(blurry_folder, exist_ok=True)

    files = list_files(input_folder)
    for ind, image_path in enumerate(files, 1):
        image_name = os.path.basename(image_path)
        print(f"[laplacian] {ind}/{len(files)} {image_name}", flush=True)

        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f" Failed to load image: {image_path}. Skipping...", flush=True)
            continue

        fm = cv2.Laplacian(downscale(gray, max_size), cv2.CV_64F).var()

        if fm < threshold:
            print(f"{image_name} is blurry (score {fm:.1f}).", flush=True)
            safe_move(image_path, blurry_folder)

    return blurry_folder


def classify_with_model(trained_model, input_folder, model_threshold=0.5):
    """Re-check images in input_folder with the model and move the blurry ones to input_folder/Blurry."""
    blurry_folder = os.path.join(input_folder, BLURRY_FOLDER_NAME)
    os.makedirs(blurry_folder, exist_ok=True)

    files = list_files(input_folder)
    for ind, image_path in enumerate(files, 1):
        image_name = os.path.basename(image_path)
        print(f"[model] {ind}/{len(files)} {image_name}", flush=True)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error reading image: {image_path}", flush=True)
            continue

        if is_image_blurry(trained_model, img, model_threshold):
            print(f"Yes, {image_name} is blurry.", flush=True)
            safe_move(image_path, blurry_folder)
        else:
            print(f"{image_name} is sharp.", flush=True)


def is_image_blurry(trained_model, img, mo_threshold=0.5):
    feature_extractor = featureExtractor()
    feature_extractor.resize_image(img, img.shape[0], img.shape[1])

    # compute the image ROI using local entropy filter
    feature_extractor.compute_roi()

    # extract the blur features using DCT transform coefficients
    extracted_features = np.array(feature_extractor.extract_feature())
    if len(extracted_features) == 0:
        return True

    # Classify all blocks of the image in one batch
    x = torch.from_numpy(extracted_features / 255.0).float().to(device)
    with torch.no_grad():
        predicted_labels = torch.argmax(trained_model(x), dim=1)

    return predicted_labels.float().mean().item() < mo_threshold


def load_model(model_path):
    trained_model = torch.load(model_path, map_location=device, weights_only=False)
    trained_model = trained_model['model_state'] if isinstance(trained_model, dict) else trained_model
    return trained_model.to(device).eval()


def process_images(input_folder, threshold, model_threshold, model_path=None, modelbased=False, max_size=0):
    # Step 1: Detect blurry images based on Laplacian variance
    blurry_folder = detect_blurry_images(input_folder, threshold, max_size)

    # Step 2: Re-classify the Laplacian results with the PyTorch model.
    # Images the model also considers blurry end up in Blurry/Blurry.
    if modelbased:
        classify_with_model(load_model(model_path), blurry_folder, model_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', required=True, help="Input folder with images")
    parser.add_argument('-t', '--threshold', default=19.0, type=float, help="Threshold for blur detection")
    parser.add_argument('-mt', '--model_threshold', default=0.5, type=float, help="Threshold for model-based classification")
    parser.add_argument('-m', '--model', help="Path to the trained PyTorch model for model-based classification")
    parser.add_argument('-mb', '--modelbased', action='store_true', help="Enable model-based classification")
    parser.add_argument('-s', '--max-size', default=0, type=int,
                        help="Downscale images so the longest side is at most this many pixels before the "
                             "Laplacian check (0 = full resolution). Faster and makes the threshold independent "
                             "of image resolution, but the threshold needs recalibrating.")

    args = parser.parse_args()

    if args.modelbased and not args.model:
        parser.error("--modelbased requires --model")

    process_images(args.images, args.threshold, args.model_threshold, args.model, args.modelbased, args.max_size)
