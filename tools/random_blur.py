import cv2
import random, os

images = os.listdir()
ksize = (20,20) # Change kernel size to control blur ammount

# Blur random images from current directory
for imageName in images:
  if (imageName[-3:] == "jpg" or imageName[-3:] == "png") and random.randint(0,1):
    image = cv2.imread(imageName)
    image = cv2.blur(image, ksize)
    cv2.imwrite(imageName[:-4] + "_blur.jpg", image)
    os.remove(imageName)
    print(f"{imageName} blurred.")
