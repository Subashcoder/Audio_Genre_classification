from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained Random Forest model and scaler
model = joblib.load('optimized_random_forest_model.pkl')  # Replace with your model's file path
scaler = joblib.load('scaler.pkl')  # Replace with your scaler file path
label_encoder = joblib.load('label_encoder.pkl')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Feature extraction function (your provided function)
def extract_desired_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract features
        features = [
            np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            np.mean(librosa.feature.rms(y=y)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            np.mean(librosa.effects.harmonic(y)),
            np.mean(librosa.effects.percussive(y)),
            librosa.beat.tempo(y=y, sr=sr)[0],
        ]

        # Extract MFCCs (1-20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_features = [np.mean(mfccs[i - 1, :]) for i in range(1, 21)]

        # Combine all features
        return np.hstack(features + mfcc_features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['audio_file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Extract features
        features = extract_desired_features(file_path)
        if features is None:
            return "Error extracting features. Please try a different audio file."
        
        # Scale the features
        features_scaled = scaler.transform([features])  # Scale the features
        
        # Predict the genre
        prediction = model.predict(features_scaled)
        GENRE = label_encoder.inverse_transform(prediction)[0]# Get the predicted genre
        
        return render_template('result.html', genre=GENRE, filename=file.filename)
    return redirect(url_for('home'))

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
