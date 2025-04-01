# Bulk Blur Detector Web GUI
Combined https://github.com/danngalann/bulk-blur-detection and https://github.com/Utkarsh-Deshmukh/Blurry-Image-Detector with a simple web GUI in Docker.

Minor changes to original scripts (no longer case sensitive file extension). Supposed to make it easier to use the blur detection on folders. 
Browse and select a folder, define a threshold value that determines "how much" blurryness is considered "blurry" and process. Choose if you want to use model based blurry detection. 
It will process a directory of pictures and autonomously move the blurry ones to a different folder.

## Files
### blur_detector.py
Check https://github.com/danngalann/bulk-blur-detection

### Docker
Use git as source in Portainer or use docker-compose.yml
Specify path where photos are stored in the compose file.
```
Exposes port 5050
```



## Requirements
 ```Docker```
