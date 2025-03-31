from flask import Flask, jsonify, send_from_directory, render_template, request, redirect, url_for
import subprocess
import os

app = Flask(__name__)

# The path to the blur detection script (from GitHub repo)
BLUR_DETECTOR_SCRIPT = '/app/blur_detector.py'

@app.route('/browse/<path:folder_path>')
def browse(folder_path):
    try:
        items = os.listdir(folder_path)
        return jsonify(items)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    # List the directories in the specified path
    base_path = '/mnt' #spec start folder to list.
    directories = os.listdir(base_path)
    return render_template('index.html', directories=directories)

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
