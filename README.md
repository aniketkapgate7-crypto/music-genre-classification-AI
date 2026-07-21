# 🎵 Music Genre Classification AI

A machine-learning application that analyzes WAV audio files and predicts their music genres.

## 🌐 Live Application

[Open the Music Genre Classification AI](https://music-genre-classification-ai.streamlit.app)

## 📂 GitHub Repository

[View the source code](https://github.com/aniketkapgate7-crypto/music-genre-classification-AI)

## ✨ Features

- Upload one or multiple WAV audio files
- Classify music into 10 genres
- Display prediction confidence
- Show the top genre predictions
- Download batch results as CSV
- Visualize audio waveforms
- Generate Mel-spectrograms

## 🎼 Supported Genres

Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae and Rock.

## 🛠️ Technologies

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## 🔄 Project Workflow

Audio upload  
→ preprocessing  
→ feature extraction  
→ trained machine-learning model  
→ genre prediction  
→ confidence and visualization

## 🧠 Extracted Audio Features

- MFCC
- Chroma
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral roll-off
- Spectral contrast
- Tempo

## ▶️ Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py