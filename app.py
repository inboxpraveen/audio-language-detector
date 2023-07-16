import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from speechbrain.pretrained import EncoderClassifier

from utils import generate_cropped_segments

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mpga', 'wav', 'mp3', 'opus', 'wma'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

language_id = EncoderClassifier.from_hparams(source="./model/")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/test_single', methods=['GET', 'POST'])
def test_single():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predictions = []
            print("file: ",file)
            print("file.filename: ",file.filename)

            segments = generate_cropped_segments(filepath)
            print(segments)
            for start, segment in enumerate(segments):
                outputs = language_id.classify_batch(segment)
                predictions.append((start*3, (start+1)*3, outputs[3][0]))
                print("predictions[start]: ", predictions[start])
                
            print(predictions)
            return render_template('single_result.html', predictions=predictions)

    return render_template('test_single.html')


@app.route('/test_directory', methods=['GET', 'POST'])
def test_directory():
    if request.method == 'POST':
        folder = request.form['folder']
        output_path = os.path.join("./output/", folder)
        os.makedirs(output_path, exist_ok=True)
        # Perform predictions on each audio file in the directory and save results to the output folder
        # Your prediction and saving logic here
        return redirect(url_for('home'))
    return render_template('test_directory.html')


if __name__ == '__main__':
    app.run(debug=True)
