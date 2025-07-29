import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# Constants
BATCH_SIZE = 64
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
VOCAB_SIZE = len(word2index)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        outputs = torch.zeros(batch_size, trg_len, VOCAB_SIZE).to(DEVICE)

        hidden = self.encoder(src)
        input = trg[:, 0]  # start with <sos>

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

# Instantiate models
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
model = Seq2Seq(encoder, decoder).to(DEVICE)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=word2index['<pad>'])
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
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
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "seq2seq_no_attention.pt")
