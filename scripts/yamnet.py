import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

SAMPLE_RATE = 16000
TARGET_LENGTH = 5 * SAMPLE_RATE  # 5 seconds

# Function to load and preprocess audio files
def load_and_preprocess(filename, augment=False):
    wav, _ = librosa.load(filename, sr=SAMPLE_RATE, mono=True)
    wav = librosa.util.normalize(wav)  # Normalize amplitude

    # Padding or truncation
    if len(wav) < TARGET_LENGTH:
        wav = np.pad(wav, (0, TARGET_LENGTH - len(wav)), 'constant')
    else:
        wav = wav[:TARGET_LENGTH]

    if augment:
        wav = apply_augmentations(wav)

    return wav

# Function to apply data augmentations
def apply_augmentations(wav):
    """Applies data augmentation techniques"""
    # Time Stretching (random rate between 0.8x and 1.2x)
    if np.random.rand() < 0.5:  
        rate = np.random.uniform(0.8, 1.2)
        wav = librosa.effects.time_stretch(wav, rate)

    # Time Shifting (random shift up to 0.5 sec)
    if np.random.rand() < 0.5:
        shift = np.random.randint(0, int(0.5 * SAMPLE_RATE))  # Max shift = 0.5 sec
        wav = np.roll(wav, shift)

    # Pitch Shifting (random pitch shift of ±2 semitones)
    if np.random.rand() < 0.5:
        semitones = np.random.uniform(-2, 2)
        wav = librosa.effects.pitch_shift(wav, sr=SAMPLE_RATE, n_steps=semitones)

    # Background Noise Addition (random noise at 5-10% of signal amplitude)
    if np.random.rand() < 0.5:
        noise = np.random.randn(len(wav)) * 0.05  # 5% noise level
        wav = wav + noise

    # Dynamic Range Compression (DRC)
    if np.random.rand() < 0.5:
        wav = librosa.effects.percussive(wav)  # Simulates DRC by reducing background energy

    return wav

# Function to process dataset in parallel
def process_dataset_parallel(file_paths, augment=False):
    """Loads and preprocesses audio files in parallel"""
    with ThreadPoolExecutor(max_workers=8) as executor:
        processed_wavs = list(executor.map(lambda f: load_and_preprocess(f, augment), file_paths))
    return processed_wavs

# Function to load the dataset
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

    data_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Save preprocessed dataset to a pickle file
    with open("dataset_cache.pkl", "wb") as f:
        pickle.dump((audio_data, labels), f)

    return tf.data.Dataset.from_tensor_slices((data_tensor, labels_tensor))

# Load the dataset from the specified directory
base_directory = "data"  # Adjust path as needed
dataset = load_dataset(base_directory)

# Function to analyze the dataset
def analyze_dataset(dataset):
    n_samples = len(dataset)
    unique_classes = set()
    for _, label in dataset:
        unique_classes.add(label.numpy())
    n_classes = len(unique_classes)
    print(f"Number of samples: {n_samples}")
    print(f"Number of classes: {n_classes}")

    # Samples per class
    samples_per_class = np.zeros(n_classes)
    for _, label in dataset:
        samples_per_class[label.numpy()] += 1
    print("Samples per class:", samples_per_class)

# Call the function to analyze the dataset
analyze_dataset(dataset)

# Function to split the dataset into train, validation, and test sets
def split_dataset(dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    n_samples = len(dataset)
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    n_test = n_samples - n_train - n_val

    dataset = dataset.shuffle(n_samples)
    train_dataset = dataset.take(n_train)
    val_dataset = dataset.skip(n_train).take(n_val)
    test_dataset = dataset.skip(n_train + n_val).take(n_test)

    return train_dataset, val_dataset, test_dataset

# Split the dataset
train_dataset, val_dataset, test_dataset = split_dataset(dataset)

# Analyze the splits
print("Train dataset:")
analyze_dataset(train_dataset)
print("\nValidation dataset:")
analyze_dataset(val_dataset)
print("\nTest dataset:")
analyze_dataset(test_dataset)

# Load preprocessed dataset (waveforms & labels)
with open("dataset_cache.pkl", "rb") as f:
    audio_data, labels = pickle.load(f)

# Convert numpy arrays to TensorFlow tensors
audio_data = tf.convert_to_tensor(audio_data, dtype=tf.float32)
labels = tf.convert_to_tensor(labels, dtype=tf.int32)

# Convert labels to categorical (one-hot encoding for 3 classes)
labels = tf.one_hot(labels, depth=3)

# Get indices for splitting
indices = np.arange(len(audio_data))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.27, stratify=tf.argmax(labels, axis=1), random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.3, 
    stratify=tf.argmax(tf.gather(labels, temp_idx), axis=1), 
    random_state=42
)

# Use tf.gather to split the data
train_audio = tf.gather(audio_data, train_idx)
train_labels = tf.gather(labels, train_idx)
val_audio = tf.gather(audio_data, val_idx)
val_labels = tf.gather(labels, val_idx)
test_audio = tf.gather(audio_data, test_idx)
test_labels = tf.gather(labels, test_idx)

print(f"Train: {len(train_audio)}, Val: {len(val_audio)}, Test: {len(test_audio)}")

# Load YAMNet pre-trained model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Function to extract features using YAMNet
def extract_features(waveform):
    """Extracts YAMNet embeddings and averages them over time."""
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return np.mean(embeddings.numpy(), axis=0)  # Take mean across time axis

# Extract features from the audio data
train_features = np.array([extract_features(w) for w in train_audio])
val_features = np.array([extract_features(w) for w in val_audio])
test_features = np.array([extract_features(w) for w in test_audio])

print(f"Fixed Train Features Shape: {train_features.shape}")  # Should be (N, 1024)

print(f"Train Features: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Build the classifier model
model = Sequential([
    Dense(256, activation='relu', input_shape=(1024,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 output classes
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(np.argmax(train_labels, axis=1)),
    y=np.argmax(train_labels, axis=1)
)

class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class Weights:", class_weight_dict)

# Train the model
history = model.fit(
    train_features, train_labels,
    validation_data=(val_features, val_labels),
    epochs=30,  # Adjust based on performance
    batch_size=32,
    class_weight=class_weight_dict
)

# Save model for later use
model.save("yamnet_classifier.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_history.png')  # Save the plot as an image

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model (if needed)
model = tf.keras.models.load_model("yamnet_classifier.h5")

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_features, test_labels)
print(f"Test Accuracy: {test_acc:.4f}")

# Predict on test samples
predictions = model.predict(test_features)
predicted_classes = np.argmax(predictions, axis=1)

# Convert one-hot labels back to class indices
true_classes = np.argmax(test_labels, axis=1)

# Print classification report
report = classification_report(true_classes, predicted_classes, target_names=["Cry", "Scream", "Speech"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df[['precision', 'recall', 'f1-score']])

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
conf_matrix_df = pd.DataFrame(conf_matrix, index=["Cry", "Scream", "Speech"], columns=["Cry", "Scream", "Speech"])
print("\nConfusion Matrix:")
print(conf_matrix_df)

# Count samples per class
cry_count = np.sum(np.argmax(train_labels, axis=1) == 0)
scream_count = np.sum(np.argmax(train_labels, axis=1) == 1)
speech_count = np.sum(np.argmax(train_labels, axis=1) == 2)

print(f"Cry: {cry_count}, Scream: {scream_count}, Speech: {speech_count}")