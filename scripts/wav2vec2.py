import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix

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

    if augment:
        wav = apply_augmentations(wav)

    return wav

def apply_augmentations(wav):
    """Applies data augmentation techniques"""
    if np.random.rand() < 0.5:  
        rate = np.random.uniform(0.8, 1.2)
        wav = librosa.effects.time_stretch(wav, rate=rate)

    if np.random.rand() < 0.5:
        shift = np.random.randint(0, int(0.5 * SAMPLE_RATE))  # Max shift = 0.5 sec
        wav = np.roll(wav, shift)

    if np.random.rand() < 0.5:
        semitones = np.random.uniform(-2, 2)
        wav = librosa.effects.pitch_shift(wav, sr=SAMPLE_RATE, n_steps=semitones)

    if np.random.rand() < 0.5:
        noise = np.random.randn(len(wav)) * 0.05  # 5% noise level
        wav = wav + noise

    if np.random.rand() < 0.5:
        wav = librosa.effects.percussive(wav)  # Simulates DRC by reducing background energy

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

    # Stack list of arrays and convert to PyTorch tensors
    data_tensor = torch.tensor(np.stack(audio_data), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    # Create and return a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
    return dataset

# Load the dataset
dataset = load_dataset("data", augment=True)

# Split the dataset into train, validation, and test sets taking care of stratification
train_size = int(0.72 * len(dataset))
val_size = int(0.18 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

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
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

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

train_embeddings, train_labels = extract_embeddings(train_loader)
val_embeddings, val_labels = extract_embeddings(val_loader)
test_embeddings, test_labels = extract_embeddings(test_loader)

# Create TensorDatasets for embeddings
train_dataset = TensorDataset(train_embeddings, train_labels)
val_dataset = TensorDataset(val_embeddings, val_labels)
test_dataset = TensorDataset(test_embeddings, test_labels)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x

mlp_model = MLP(input_dim=train_embeddings.shape[1], output_dim=3)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=3e-5)

# Variables to store losses and accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_accuracy = 0.0
best_model_state = None

# Train the MLP model
num_epochs = 40
for epoch in range(num_epochs):
    mlp_model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for inputs, labels in tqdm(DataLoader(train_dataset, batch_size=8, shuffle=True), desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        outputs = mlp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
    train_losses.append(train_loss / len(train_dataset))
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Validation
    mlp_model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in tqdm(DataLoader(val_dataset, batch_size=8, shuffle=False), desc=f"Validation Epoch {epoch+1}"):
            outputs = mlp_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted_val = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()
    val_losses.append(val_loss / len(val_dataset))
    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = mlp_model.state_dict()

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_dataset)}, Train Accuracy: {train_accuracy}, Validation Loss: {val_loss/len(val_dataset)}, Validation Accuracy: {val_accuracy}")

# Save the best model
torch.save(best_model_state, "best_mlp_model.pth")

# Plot the losses and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epochs')

# Save the plot instead of showing it
plt.savefig("training_validation_metrics.png")

# Evaluate on test set: precision, accuracy, f1 score, confusion matrix
best_model = MLP(input_dim=test_embeddings.shape[1], output_dim=3)
best_model.load_state_dict(torch.load("best_mlp_model.pth"))
best_model.eval()

# Get predictions
preds = []
with torch.no_grad():
    for inputs, labels in DataLoader(test_dataset, batch_size=8, shuffle=False):
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        preds.extend(predicted.tolist())

# Compute metrics
precision = precision_score(test_labels, preds, average='weighted')
accuracy = accuracy_score(test_labels, preds)
f1 = f1_score(test_labels, preds, average='weighted')
conf_matrix = confusion_matrix(test_labels, preds)

# Print the metrics
print(f"Precision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)