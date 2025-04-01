from flask import Flask, jsonify, render_template, request, redirect, url_for
import subprocess
import os

app = Flask(__name__)

MODEL_PATH = '/app/model/trained_model-Kaggle_dataset'  # Update this to the correct model path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    source_folder = request.form['source_folder']
    threshold = request.form['threshold']
    model_based = 'modelbased' in request.form  # Checkbox value
    final_blurry_folder = request.form.get('final_blurry_folder', '')
    
    # Validate the source folder and threshold
    if not os.path.exists(source_folder):
        return "Source folder does not exist!", 400
    if not threshold.isdigit() or int(threshold) < 0:
        return "Invalid threshold!", 400

    # Run the blur detector script with user parameters
    command = [
        'python', 
        '/app/process_blurry_images.py', 
        '-i', source_folder,           # input folder containing images to process
        '-t', str(threshold),          # threshold for Laplacian blurriness detection
        '-f', final_blurry_folder,     # final folder to move the detected blurry images
        '-m', MODEL_PATH              # path to the pre-trained model
    ]
    
    # If model-based classification is enabled
    if model_based:
        command.append('-m')  # Add model path to the command

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)  # Optionally log the output for debugging
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")  # Log the error message
        return f"An error occurred while processing images: {e.stderr}", 500

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
