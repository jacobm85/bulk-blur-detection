from flask import Flask, jsonify, send_from_directory, render_template, request, redirect, url_for
import subprocess
import os
from flask_socketio import SocketIO, emit
import eventlet

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# BLUR_DETECTOR_SCRIPT = '/app/blur_detector.py'
BLUR_DETECTOR_SCRIPT = '/app/process_blurry_images.py'
BASE_DIR = '/app/images'  # Change this to your desired path

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('browse')
def browse(data):
    path = os.path.join(BASE_DIR, data.get('path', ''))
    if not os.path.exists(path):
        emit('error', {'message': 'Directory does not exist!'})
        return

    files = os.listdir(path)
    directories = [{'name': f, 'is_dir': os.path.isdir(os.path.join(path, f))} for f in files]
    emit('files', {'path': path, 'files': directories})

@app.route('/process', methods=['POST'])
def process_images():
    source_folder = request.form['source_folder']
    threshold = request.form['threshold']
    use_model = 'use_model' in request.form  # Check if the user checked the checkbox

    # Validate the source folder and threshold
    if not os.path.exists(source_folder):
        return "Source folder does not exist!", 400
    if not threshold.isdigit() or int(threshold) < 0:
        return "Invalid threshold!", 400

    # Run the blur detector script with user parameters
    # command = ['python', BLUR_DETECTOR_SCRIPT, '-i', source_folder, '-t', threshold]

    # Define the command to run the script based on the chec
    command = [
        'python', 
        BLUR_DETECTOR_SCRIPT, 
        '-i', source_folder,          # input folder containing images to process
        '-t', str(threshold),         # threshold for Laplacian blurriness detection
        '-m', trained_model_path      # path to the pre-trained model
    ]

    # If the user wants to use the trained model, add extra argument
    if use_model:
        command.append('-f')
        command.append(final_blurry_folder)  # Add final blurry folder

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)  # Optionally log the output for debugging
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")  # Log the error message
        return f"An error occurred while processing images: {e.stderr}", 500

    return redirect(url_for('index'))

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
