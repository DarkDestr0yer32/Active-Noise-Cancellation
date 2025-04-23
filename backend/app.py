from flask import Flask, request, jsonify, send_file, render_template
from .utils.predict import denoise_audio
from .utils.visualize import generate_all_plots
import os
import time

# Initialize Flask app
app = Flask(__name__)

# Create the upload folder dynamically if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.getcwd(), "backend/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to serve the homepage
@app.route('/')
def index():
    return render_template("index.html")

# Route to handle audio file uploads
@app.route('/upload', methods=['POST'])
def upload_audio():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the uploaded file (denoise and generate plots)
    denoised_path, noisy_audio, denoised_audio, chunk_results = denoise_audio(filepath)
    plot_paths = generate_all_plots(filepath, denoised_audio, noisy_audio, chunk_results)

    # Return the denoised audio path and plot file names
    return jsonify({
        'denoised_audio': os.path.basename(denoised_path),
        'plots': [os.path.basename(p) for p in plot_paths]
    })

# Route to serve audio files and plots
@app.route('/get_audio/<path:filename>')
def get_audio(filename):
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    for _ in range(5):
        if os.path.exists(full_path):
            return send_file(full_path)
        time.sleep(0.2)
    return f"File not found: {filename}", 404

# Run the app in debug mode if executed directly
if __name__ == '__main__':
    app.run(debug=True)
