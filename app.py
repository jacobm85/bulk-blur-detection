from flask import Flask, jsonify, send_from_directory, render_template, request, redirect, url_for
import subprocess
import os

app = Flask(__name__)

# The path to the blur detection script (from GitHub repo)
BLUR_DETECTOR_SCRIPT = '/app/blur_detector.py'
# Set the directory you want to browse
BASE_DIR = '/'  # Change this to your desired path

@app.route('/', methods=['GET', 'POST'])
def index():
    current_path = request.args.get('path', BASE_DIR)
    
    if not os.path.exists(current_path):
        return "Directory does not exist", 404
    
    # List directories and files
    items = os.listdir(current_path)
    directories = [item for item in items if os.path.isdir(os.path.join(current_path, item))]
    files = [item for item in items if os.path.isfile(os.path.join(current_path, item))]

    return render_template('index.html', directories=directories, files=files, current_path=current_path)

@app.route('/browse', methods=['POST'])
def browse():
    folder = request.form.get('folder')
    return index(path=os.path.join(request.args.get('path', BASE_DIR), folder))

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
    app.run(debug=True, host='0.0.0.0', port=5000)
