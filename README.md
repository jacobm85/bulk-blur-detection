# Bulk Blur Detector Web GUI
Docker version with a simple web GUI of https://github.com/danngalann/bulk-blur-detection. 
Browse to and select a folder, define a threshold value that determines "how much" blurryness is considered "blurry" and process. No changes made the the executed script, just supposed to make it easier to use it. 
It will process a directory of pictures and autonomously move the blurry ones to a different folder.

## Files
### blur_detector.py
Check https://github.com/danngalann/bulk-blur-detection

### Docker
Use git as source in Portainer or use docker-compose.yml
Specify path where photos are stored in the compose file.
```
Exposes port 5000
```



## Requirements
 ```Docker```
