import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mpga', 'wav', 'mp3', 'opus', 'wma'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/test_single', methods=['GET', 'POST'])
def test_single():

    if request.method == 'POST':
        audio_file = request.files['audio_file']
        if audio_file and allowed_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)

            predictions = []
            print("file: ",audio_file)
            print("file_path: ",filepath)

            # segments = preprocess_audio(filepath)
            # print(segments)
            # for start, segment in enumerate(segments):
            #     print("segment: ",segment)
            #     print("start: ",start)
            #     outputs = language_id.classify_batch(language_id.load_audio(segment))
            #     predictions.append((start*3, (start+1)*3, outputs[3][0]))
            #     print("predictions[start]: ", predictions[start])
            predictions.append(language_id.load_audio("./uploads/Recording.mp3"))
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
