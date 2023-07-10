import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mpga', 'wav', 'mp3', 'opus', 'wma'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        folder = request.form['folder']
        # Perform training on the folder containing audio files
        # Your training logic here
        return redirect(url_for('home'))
    return render_template('train.html')


@app.route('/test_single', methods=['GET', 'POST'])
def test_single():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Perform predictions on the single audio file
            # Your prediction logic here
            predictions = [(0, 10, 'English'), (12, 22, 'French'), (25, 30, 'Spanish')]  # Replace with actual predictions
            return render_template('single_result.html', predictions=predictions)
    return render_template('test_single.html')


@app.route('/test_directory', methods=['GET', 'POST'])
def test_directory():
    if request.method == 'POST':
        folder = request.form['folder']
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], folder)
        os.makedirs(output_path, exist_ok=True)
        # Perform predictions on each audio file in the directory and save results to the output folder
        # Your prediction and saving logic here
        return redirect(url_for('home'))
    return render_template('test_directory.html')


if __name__ == '__main__':
    app.run(debug=True)
