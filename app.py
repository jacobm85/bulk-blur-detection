from flask import Flask, jsonify, send_from_directory, render_template, request, redirect, url_for
import subprocess
import os
from flask_socketio import SocketIO, emit
import eventlet

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Path to blur detector script and other constants
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
    #model_based = request.form.get('model_based') == 'on'  # On if checked, False otherwise
    #model_based = request.form.get('model_based', False)
    model_based = request.form.get('modelbased') == 'True'

    # Debugging log: print the received source folder
    print(f"Received source folder: {source_folder}")

    # Validate the source folder and threshold
    if not os.path.exists(source_folder):
        return f"Source folder does not exist: {source_folder}!", 400

    if not threshold.isdigit() or int(threshold) < 0:
        return "Invalid threshold!", 400

    # Run the blur detector script with user parameters
    command = [
        'python', 
        BLUR_DETECTOR_SCRIPT, 
        '-i', source_folder,          # input folder containing images to process
        '-t', str(threshold),         # threshold for Laplacian blurriness detection
        '-m', '/app/trained_model/trained_model-Kaggle_dataset',  # Correct path to the model for classification
        #'-m', '/app/trained_model/trained_model-BSD-B',  # Correct path to the model for classification
    ]

    if model_based:
        command.append('-mb')  # Add the '-mb' flag if model-based classification is enabled
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)  # Log the output from the model-based classification process
        return jsonify({"message": "Processing completed successfully"})
    except subprocess.CalledProcessError as e:
        print(f"Model classification error: {e.stderr}")
        return f"Error in model classification: {e.stderr}", 500

    return redirect(url_for('index'))

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
