import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import torch
import soundfile as sf
import io
import tensorflow_hub as hub
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import torch.nn as nn

SAMPLE_RATE = 16000
TARGET_LENGTH = 5 * SAMPLE_RATE  # 5 seconds
LABEL_MAP = {"cry": 0, "scream": 1, "speech": 2}

def load_and_preprocess_chunk(audio_chunk):
    # Read the audio chunk using soundfile
    wav, _ = sf.read(io.BytesIO(audio_chunk), samplerate=SAMPLE_RATE, channels=1, format='RAW', subtype='PCM_16')

    return wav

def preprocess_and_save_wav(filepath):
    # Load the audio file
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    # Normalize amplitude
    audio = librosa.util.normalize(audio)

    # Save the preprocessed audio as a new WAV file
    processed_filepath = filepath.replace(".wav", "_processed.wav")
    sf.write(processed_filepath, audio, samplerate=SAMPLE_RATE)

    return processed_filepath

def load_dataset_from_chunk(audio_data):
    """Loads and preprocesses audio chunks and generates dummy labels"""

    yamnet_data_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)

    # Stack list of arrays and convert to PyTorch tensors
    data_tensor = torch.tensor(np.stack(audio_data), dtype=torch.float32)

    # Create and return a PyTorch dataset
    wav2vec2_dataset = torch.utils.data.TensorDataset(data_tensor)

    return yamnet_data_tensor, wav2vec2_dataset

def convert_to_chunk(filename: str):
    # Load the .wav file
    wav, _ = librosa.load(filename, sr=SAMPLE_RATE, mono=True)

    # Convert to audio bytes
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, wav, SAMPLE_RATE, format='WAV')
    audio_bytes.seek(0)

    return audio_bytes.read()

def classify_wrapper(filepath):
    audio_chunk = load_and_preprocess_chunk(convert_to_chunk(filepath))
    return classify_audio_using_ensemble(audio_chunk)

def classify_audio_using_ensemble(audio_data):
    yamnet_data_tensor, wav2vec2_dataset = load_dataset_from_chunk(audio_data)

    # Load the YAMNet pre-trained model
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    def extract_features(waveform):
        """Extract YAMNet embeddings and average them over time."""
        scores, embeddings, spectrogram = yamnet_model(waveform)
        # Use tf.reduce_mean and then convert to NumPy if needed
        return tf.reduce_mean(embeddings, axis=0).numpy()

    # Extract features for the single waveform
    test_features_yamnet = extract_features(yamnet_data_tensor)

    # Load the model
    wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    # Extract embeddings
    def extract_embeddings(tensor):
        wav2vec2_model.eval()
        with torch.no_grad():
            outputs = wav2vec2_model(tensor.unsqueeze(0)).last_hidden_state.mean(dim=1)  # Mean pooling
        return outputs

    wav2vec2_embeddings = extract_embeddings(wav2vec2_dataset.tensors[0])

    # Define the MLP model
    class wav2vec2_MLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(wav2vec2_MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )

        def forward(self, x):
            x = self.model(x)
            return x

    # Load trained models
    yamnet_model = tf.keras.models.load_model("model_weights/best_yamnet.h5")
    wav2vec2_mlp = wav2vec2_MLP(input_dim=wav2vec2_embeddings.shape[1], output_dim=3)
    wav2vec2_mlp.load_state_dict(torch.load("model_weights/best_mlp_wav2vec2.pth"))

    # Get prediction probabilities for each model
    yamnet_probs = yamnet_model.predict(np.expand_dims(test_features_yamnet, axis=0))
    wav2vec2_mlp.eval()
    with torch.no_grad():
        wav2vec2_probs = torch.nn.functional.softmax(wav2vec2_mlp(wav2vec2_embeddings), dim=1)
        wav2vec2_probs = wav2vec2_probs.numpy()  # Convert to NumPy (Shape: (1, num_classes))

    # Ensemble: Simple averaging
    ensemble_probs = (yamnet_probs + wav2vec2_probs) / 2
    ensemble_predictions = np.argmax(ensemble_probs, axis=1)

    # Convert predictions to labels
    label_map = {v: k for k, v in LABEL_MAP.items()}
    final_predictions = [label_map[pred] for pred in ensemble_predictions]

    # Return results as a dictionary with label and confidence score of the predicted class
    return {"label": final_predictions[0], "confidence": ensemble_probs[0][ensemble_predictions[0]]}