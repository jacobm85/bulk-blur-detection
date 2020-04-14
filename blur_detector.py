import cv2
import argparse, os

# Compute the Laplacian of the image and then return the focus measure
def variance_of_laplacian(image):  
  return cv2.Laplacian(image, cv2.CV_64F).var()

# Construct arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
  help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
  help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

# Parse arguments
imageDir = args["images"]
threshold = args["threshold"]

# loop over the input images
for imageName in os.listdir(imageDir):
  # If the current path is not an image, skip this iteration
  if imageName[-3:] != "jpg" and imageName[-3:] != "png":
    continue

  # Initialize paths for I/O
  imagePath = os.path.join(imageDir, imageName)
  blurryPath = os.path.join(imageDir, "blurry")

  # load the image, convert it to grayscale, and compute the
  # focus measure of the image
  image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  fm = variance_of_laplacian(gray)

  # if the image blurryness is above the threshold, write the image to a "blurry" folder
  if fm < threshold:
    print(imageName + " is blurry.")
    # If the "blurry" folder doesn't exists, create it
    if not os.path.exists(blurryPath):
      os.mkdir(blurryPath)
    cv2.imwrite(os.path.join(blurryPath, imageName), image)
    os.remove(imagePath) # Remove original so it's only on the "blurry" folder