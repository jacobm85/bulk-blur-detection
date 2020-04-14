# Bulk Blur Detector
This script is based on a project from (pyimagesearch)[https://www.pyimagesearch.com]. It will process a directory of pictures and autonomously move the blurry ones to a different folder.
This is specially useful to me as a hobbist photographer, so I can quickly discard blurry pictures without having to manually check every one of them.

## Files
### blur_detector.py
This is the main script. It will take as arguments the directory of the pictures to be precessed, and optionally a threshold value that determines "how much" blurryness is considered "blurry".

```
usage: blur_detector.py [-h] -i IMAGES [-t THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGES, --images IMAGES
                        path to input directory of images
  -t THRESHOLD, --threshold THRESHOLD
                        focus measures that fall below this value will be
                        considered 'blurry'
```

### local_blur_detector.py
This is a local version of the main script. You can drop it directly to the target directory and execute it without worring about the arguments. This makes it easier to use to regular users.
Note that in this version of the script, you can't alter the threshold.

### tools/random_blur.py
This is a simple script to blur random images from a directory. It's meant to be used for tests.

## Requirements
This script requires Python OpenCV, you can install it through ```pip```
```pip install opencv-python```