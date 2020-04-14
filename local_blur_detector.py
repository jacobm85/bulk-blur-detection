import cv2, os

# Compute the Laplacian of the image and then return the focus measure
def variance_of_laplacian(image):  
  return cv2.Laplacian(image, cv2.CV_64F).var()

# Set threshold
threshold = 100.0

# loop over the input images
for imageName in os.listdir():
  # If the current path is not an image, skip this iteration
  if imageName[-3:] != "jpg" and imageName[-3:] != "png":
    continue

  # load the image, convert it to grayscale, and compute the
  # focus measure of the image
  image = cv2.imread(imageName)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  fm = variance_of_laplacian(gray)

  # if the image blurryness is above the threshold, write the image to a "blurry" folder
  if fm < threshold:
    print(imageName + " is blurry.")
    # If the "blurry" folder doesn't exists, create it
    if not os.path.exists("blurry"):
      os.mkdir("blurry")
    cv2.imwrite(os.path.join("blurry", imageName), image)
    os.remove(imageName) # Remove original so it's only on the "blurry" folder