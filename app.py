from flask import Flask, jsonify, send_from_directory, render_template, request, redirect, url_for
import os
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

BASE_DIR = '/app/images'  # This will be the root directory for browsing images

@app.route('/')
def index():
    return render_template('index.html')

# Handle browsing through directories and show only directories
@socketio.on('browse')
def browse(data):
    # Default to /app/images if no path is provided
    path = os.path.join(BASE_DIR, data.get('path', '')) if data.get('path') else BASE_DIR
    if not os.path.exists(path):
        emit('error', {'message': 'Directory does not exist!'})
        return

    files = os.listdir(path)
    directories = [{'name': f, 'is_dir': os.path.isdir(os.path.join(path, f))} for f in files if os.path.isdir(os.path.join(path, f))]
    emit('files', {'path': path, 'files': directories})

# Route for processing images (your existing process function)
@app.route('/process', methods=['POST'])
def process_images():
    source_folder = request.form['source_folder']
    threshold = request.form['threshold']
    model_based = 'modelbased' in request.form  # Check if model-based classification is selected

    # Validate the source folder and threshold
    if not os.path.exists(source_folder):
        return "Source folder does not exist!", 400
    if not threshold.isdigit() or int(threshold) < 0:
        return "Invalid threshold!", 400

    # Determine blurry folder location
    blurry_folder = os.path.join(source_folder, "Blurry")
    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)

    # Run the blur detection logic (add your detection functions here)
    detect_blurry_images(source_folder, threshold)

    # If the user checked the checkbox for model-based classification, run it
    if model_based:
        model_path = request.form['model_path']
        model_based_classification(blurry_folder, model_path)

    return redirect(url_for('index'))

# Function for detecting blurry images
def detect_blurry_images(input_folder, threshold=100.0):
    blurry_folder = os.path.join(input_folder, "Blurry")
    if not os.path.exists(blurry_folder):
        os.mkdir(blurry_folder)

    # Implement your logic to detect blurry images and move them to the "Blurry" folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):  # Ensure it is a file
            # Assuming a function `is_blurry(image_path)` exists
            if is_blurry(image_path, threshold):
                os.rename(image_path, os.path.join(blurry_folder, image_name))  # Move the file

# Function for model-based classification
def model_based_classification(input_folder, model_path):
    blurry_folder = os.path.join(input_folder, "Blurry")
    blurry_folder_new = os.path.join(blurry_folder, "Blurry")

    if not os.path.exists(blurry_folder_new):
        os.mkdir(blurry_folder_new)

    # Loop through images in the "Blurry" folder and classify using the model
    for image_name in os.listdir(blurry_folder):
        image_path = os.path.join(blurry_folder, image_name)
        if os.path.isfile(image_path):
            # Assuming a function `classify_image_with_model(image_path, model_path)` exists
            if classify_image_with_model(image_path, model_path):
                os.rename(image_path, os.path.join(blurry_folder_new, image_name))  # Move the image to the new folder

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
