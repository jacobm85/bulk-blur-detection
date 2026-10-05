import eventlet
eventlet.monkey_patch()  # Must run before other imports

import os
import sys

from eventlet import tpool
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Use the real (non-green) subprocess module; its output is read in a native thread via tpool
# so the server keeps responding while a job runs, on every platform.
subprocess = eventlet.patcher.original('subprocess')

# Path to blur detector script and other constants
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BLUR_DETECTOR_SCRIPT = os.path.join(APP_DIR, 'process_blurry_images.py')
MODEL_PATH = os.path.join(APP_DIR, 'trained_model', 'trained_model-Kaggle_dataset')
# MODEL_PATH = os.path.join(APP_DIR, 'trained_model', 'trained_model-BSD-B')
BASE_DIR = os.path.realpath(os.environ.get('IMAGES_DIR', '/app/images'))

current_job = None


def resolve_path(path):
    """Return the real path if it is inside BASE_DIR, otherwise None."""
    full_path = os.path.realpath(os.path.join(BASE_DIR, path or ''))
    if os.path.commonpath([full_path, BASE_DIR]) != BASE_DIR:
        return None
    return full_path


def parse_float(value, minimum, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < minimum or (maximum is not None and number > maximum):
        return None
    return number


@app.route('/')
def index():
    return render_template('index.html', base_dir=BASE_DIR)


@socketio.on('browse')
def browse(data):
    path = resolve_path(data.get('path'))
    if path is None or not os.path.isdir(path):
        emit('error', {'message': 'Directory does not exist!'})
        return

    files = sorted(os.listdir(path))
    directories = [{'name': f, 'is_dir': os.path.isdir(os.path.join(path, f))} for f in files]
    emit('files', {'path': path, 'is_root': path == BASE_DIR, 'files': directories})


@socketio.on('process')
def process_images(data):
    global current_job
    if current_job is not None:
        emit('error', {'message': 'A job is already running.'})
        return

    source_folder = resolve_path(data.get('source_folder'))
    threshold = parse_float(data.get('threshold'), 0)
    model_threshold = parse_float(data.get('model_threshold', 0.5), 0, 1)
    max_size = parse_float(data.get('max_size', 0), 0)
    model_based = bool(data.get('modelbased'))

    if source_folder is None or not os.path.isdir(source_folder):
        emit('error', {'message': 'Source folder does not exist!'})
        return
    if threshold is None:
        emit('error', {'message': 'Invalid threshold!'})
        return
    if model_threshold is None:
        emit('error', {'message': 'Invalid model threshold (must be between 0 and 1)!'})
        return
    if max_size is None:
        emit('error', {'message': 'Invalid max image size!'})
        return

    command = [
        sys.executable, '-u',
        BLUR_DETECTOR_SCRIPT,
        '-i', source_folder,          # input folder containing images to process
        '-t', str(threshold),         # threshold for Laplacian blurriness detection
        '-m', MODEL_PATH,             # model used for model-based classification
        '-mt', str(model_threshold),  # threshold for model-based classification
        '-s', str(int(max_size)),     # downscale before the Laplacian check (0 = off)
    ]
    if model_based:
        command.append('-mb')  # Add the '-mb' flag if model-based classification is enabled

    print(f"Starting job: {command}")
    current_job = socketio.start_background_task(run_job, command)
    emit('started', {'source_folder': source_folder})


def run_job(command):
    global current_job
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(lambda: tpool.execute(process.stdout.readline), ''):
            line = line.rstrip()
            print(line)
            socketio.emit('progress', {'line': line})
        tpool.execute(process.wait)
        if process.returncode == 0:
            socketio.emit('done', {'message': 'Processing completed successfully'})
        else:
            socketio.emit('done', {'message': f'Processing failed (exit code {process.returncode})', 'failed': True})
    finally:
        current_job = None


@socketio.on('connect')
def on_connect():
    emit('status', {'running': current_job is not None})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
