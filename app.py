from flask import Flask, jsonify, send_from_directory, render_template, request, redirect, url_for
import subprocess
import os
from flask_socketio import SocketIO, emit
import eventlet

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

BLUR_DETECTOR_SCRIPT = '/app/blur_detector.py'
BASE_DIR = '/app/images'  # Change this to your desired path

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('browse')
def browse(data):
    path = data.get('path', BASE_DIR)
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

    # Validate the source folder and threshold
    if not os.path.exists(source_folder):
        return "Source folder does not exist!", 400
    if not threshold.isdigit() or int(threshold) < 0:
        return "Invalid threshold!", 400

    # Run the blur detector script with user parameters
    command = ['python', BLUR_DETECTOR_SCRIPT, '-i', source_folder, '-t', threshold]
    subprocess.run(command, check=True)

    return redirect(url_for('index'))

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
