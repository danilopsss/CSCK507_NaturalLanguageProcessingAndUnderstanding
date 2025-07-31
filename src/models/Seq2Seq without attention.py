import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import sys
import os

# Add the src directory to Python path so we can import from other modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

# Import from the data_utils file in the same directory (src/models/data_utils.py)
from data_utils import load_preprocessed_data, get_vocab_size, print_data_summary

# Import from the utils package directory (src/utils/device_config.py)
from utils.device_config import get_device

device = get_device()

# Constants
BATCH_SIZE = 64
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
# VOCAB_SIZE will be defined after loading vocabulary


# Load all preprocessed data (requires notebook to be run first)
X_train, y_train, X_val, y_val, X_test, y_test, word2index, index2word = (
    load_preprocessed_data()
)

# Set vocabulary size
VOCAB_SIZE = get_vocab_size(word2index)

# Print summary
print_data_summary(X_train, y_train, X_val, y_val, X_test, y_test, word2index)


# Custom Dataset
class QADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# DataLoaders
train_dataset = QADataset(X_train, y_train)
val_dataset = QADataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.rnn = nn.GRU(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return hidden


# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.rnn = nn.GRU(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden


# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(device)

        hidden = self.encoder(src)
        input = trg[:, 0]  # start with <sos>

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs


# Instantiate models (now that VOCAB_SIZE is defined)
encoder = Encoder().to(device)
decoder = Decoder().to(device)
model = Seq2Seq(encoder, decoder).to(device)

print(f"\nModel initialized on {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=word2index["<pad>"])
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)

        # Reshape
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

# Create Models directory if it doesn't exist
import os

# Get absolute path to project root and models directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
models_dir = os.path.join(project_root, "models")

os.makedirs(models_dir, exist_ok=True)

# Save model to match evals.py expectations
model_path = os.path.join(models_dir, "chatbot_model_no_attention.pth")
torch.save(model.state_dict(), model_path)
print(f"\n✓ Model saved to: {model_path}")
