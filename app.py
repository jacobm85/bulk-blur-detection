from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return open('index.html').read()

@app.route('/process', methods=['POST'])
def process():
    input_folder = request.form['inputFolder']
    output_folder = request.form['outputFolder']
    threshold = request.form['threshold']

    # Call the blur_detector.py script with the provided arguments
    subprocess.run(['python', 'blur_detector.py', '-i', input_folder, '-t', threshold, '-o', output_folder])

    return "Processing complete!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
