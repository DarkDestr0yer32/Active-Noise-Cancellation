# app.py
from flask import Flask, request, jsonify, send_file, render_template
from utils.predict import denoise_audio
from utils.visualize import generate_all_plots
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath("backend/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_audio():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    denoised_path, noisy_audio, denoised_audio, chunk_results = denoise_audio(filepath)
    plot_paths = generate_all_plots(filepath, denoised_audio, noisy_audio, chunk_results)

    return jsonify({
        'denoised_audio': os.path.basename(denoised_path),
        'plots': [os.path.basename(p) for p in plot_paths]
    })

@app.route('/get_audio/<path:filename>')
def get_audio(filename):
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    for _ in range(5):
        if os.path.exists(full_path):
            return send_file(full_path)
        time.sleep(0.2)
    return f"File not found: {filename}", 404

if __name__ == '__main__':
    app.run(debug=True)
