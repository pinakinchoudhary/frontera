import os
import librosa
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
import torch.optim as optim

SAMPLE_RATE = 16000
TARGET_LENGTH = 5 * SAMPLE_RATE  # 5 seconds

def load_and_preprocess(filename, augment=False):
    wav, _ = librosa.load(filename, sr=SAMPLE_RATE, mono=True)
    wav = librosa.util.normalize(wav)  # Normalize amplitude

    # Padding or truncation
    if len(wav) < TARGET_LENGTH:
        wav = np.pad(wav, (0, TARGET_LENGTH - len(wav)), 'constant')
    else:
        wav = wav[:TARGET_LENGTH]

    return wav

def process_dataset_parallel(file_paths, augment=False):
    """Loads and preprocesses audio files in parallel"""
    with ThreadPoolExecutor(max_workers=8) as executor:
        processed_wavs = list(executor.map(lambda f: load_and_preprocess(f, augment), file_paths))
    return processed_wavs

def load_dataset(base_dir, augment=False):

    file_paths, labels = [], []
    label_map = {"cry": 0, "scream": 1, "speech": 2}

    for subdir, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".wav"):
                file_paths.append(os.path.join(subdir, file))
                labels.append(label_map[os.path.basename(subdir)])

    # Load and preprocess in parallel
    audio_data = process_dataset_parallel(file_paths, augment=augment)

    # Ensure the mapping of tensor to label is the same for both YAMNet and PyTorch tensors
    yamnet_data_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    yamnet_labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Stack list of arrays and convert to PyTorch tensors
    data_tensor = torch.tensor(np.stack(audio_data), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    # Create and return a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)

    return (yamnet_data_tensor, yamnet_labels_tensor), dataset

base_directory = "test_data"  # Adjust path as needed
(data, labels), dataset = load_dataset(base_directory)

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

# Load the YAMNet pre-trained model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_features(waveform):
    """Extract YAMNet embeddings and average them over time."""
    scores, embeddings, spectrogram = yamnet_model(waveform)
    # Use tf.reduce_mean and then convert to NumPy if needed
    return tf.reduce_mean(embeddings, axis=0).numpy()

test_labels_yamnet = tf.one_hot(labels, depth=3)

# Extract features for each waveform
test_features_yamnet = np.array([extract_features(waveform) for waveform in data])

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Define the data collator
def data_collator(batch):
    input_values = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    return {"input_values": input_values_padded, "labels": labels}

# Create DataLoaders
data_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

# Extract embeddings
def extract_embeddings(dataloader):
    embeddings = []
    labels = []
    wav2vec2_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"]
            label = batch["labels"]
            outputs = wav2vec2_model(input_values).last_hidden_state.mean(dim=1)  # Mean pooling
            embeddings.append(outputs)
            labels.append(label)
    return torch.cat(embeddings), torch.cat(labels)

wav2vec2_embeddings, wav2vec2_labels = extract_embeddings(data_loader)

# Create TensorDataset
wav2vec2_dataset = TensorDataset(wav2vec2_embeddings, wav2vec2_labels)

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

# Get predictions by best model from file
wav2vec2_mlp = wav2vec2_MLP(input_dim=wav2vec2_embeddings.shape[1], output_dim=3)
wav2vec2_mlp.load_state_dict(torch.load("best_mlp_wav2vec2.pth"))

# Load trained model (if needed)
yamnet_model = tf.keras.models.load_model("best_yamnet.h5")

# Get prediction probabilities for each model
yamnet_probs = yamnet_model.predict(test_features_yamnet)
wav2vec2_mlp.eval()
with torch.no_grad():
    wav2vec2_probs = torch.nn.functional.softmax(wav2vec2_mlp(wav2vec2_embeddings), dim=1)
    wav2vec2_probs = wav2vec2_probs.numpy()  # Convert to NumPy (Shape: (num_samples, num_classes))

# ENSEMBLE: 1
# Simple averaging
ensemble_probs = (yamnet_probs + wav2vec2_probs) / 2

ensemble_predictions = np.argmax(ensemble_probs, axis=1)

# Calculate precision, recall, and F1 score, confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
test_labels_yamnet_class = np.argmax(test_labels_yamnet, axis=1)

print(classification_report(test_labels_yamnet_class, ensemble_predictions))
print(confusion_matrix(test_labels_yamnet_class, ensemble_predictions))

# ENSEMBLE: 2
# Hard voting
yamnet_predictions = np.argmax(yamnet_probs, axis=1)
wav2vec2_predictions = np.argmax(wav2vec2_probs, axis=1)

# Majority voting
ensemble_predictions = np.array([np.argmax(np.bincount([y, w])) for y, w in zip(yamnet_predictions, wav2vec2_predictions)])

print(classification_report(test_labels_yamnet_class, ensemble_predictions))
print(confusion_matrix(test_labels_yamnet_class, ensemble_predictions))

# ENSEMBLE: 3
# Yamnet model was more accurate, so we will give it more weight
# Weighted averaging
ensemble_probs = (0.7 * yamnet_probs + 0.3 * wav2vec2_probs)

ensemble_predictions = np.argmax(ensemble_probs, axis=1)

# Calculate precision, recall, and F1 score, confusion matrix
print(classification_report(test_labels_yamnet_class, ensemble_predictions))
print(confusion_matrix(test_labels_yamnet_class, ensemble_predictions))
