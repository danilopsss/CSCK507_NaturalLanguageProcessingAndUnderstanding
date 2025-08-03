"""
Consolidated model utilities for Seq2Seq training and inference
Combines data loading, model architectures, and training functions
"""

import os
import random
import pickle
import ast
import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
from .device_config import get_device

# Use spaCy tokenization like in preprocessing class
spacy_en = spacy.load("en_core_web_lg")
device = get_device()

# Model hyperparameters and special tokens
SOS_TOKEN = "<sos>"
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# DATA LOADING UTILITIES


def load_preprocessed_data(base_path=None):
    """
    Load preprocessed data created by the jupyter notebook
    Returns the same variables that the notebook creates

    Prerequisites: Run preprocessing_and_split.ipynb first!
    """
    print("Loading preprocessed data...")

    # Get the path to the data folder relative to the project root
    if base_path is None:
        # Get the directory of this script (src/utils/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels to project root, then into data
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(project_root, "data")

    print(f"Looking for data in: {base_path}")

    # Check if preprocessed data exists
    processed_file = os.path.join(base_path, "DATASET_PROCESSED.csv")
    vocab_word2index = os.path.join(base_path, "word2index.pkl")
    vocab_index2word = os.path.join(base_path, "index2word.pkl")

    if not os.path.exists(processed_file):
        raise FileNotFoundError(
            "ERROR: Preprocessed data not found!\n\n"
            "Missing: {processed_file}\n\n"
            "Please run the 'preprocessing_and_split.ipynb' notebook first to create "
            "the preprocessed data.\n"
            "The notebook will create:\n"
            "  - DATASET_PROCESSED.csv\n"
            "  - word2index.pkl\n"
            "  - index2word.pkl\n\n"
            "Then you can run this training script."
        )

    if not os.path.exists(vocab_word2index) or not os.path.exists(vocab_index2word):
        raise FileNotFoundError(
            f"ERROR: Vocabulary files not found!\n\n"
            f"Missing: {vocab_word2index} or {vocab_index2word}\n\n"
            f"Please run the 'preprocessing_and_split.ipynb' notebook first."
        )

    print("Loading vocabulary...")
    with open(vocab_word2index, "rb") as f:
        word2index = pickle.load(f)
    with open(vocab_index2word, "rb") as f:
        index2word = pickle.load(f)

    print("Loading processed dataset...")
    qa_df = pd.read_csv(processed_file)
    qa_df["question_padded"] = qa_df["question_padded"].apply(ast.literal_eval)
    qa_df["answer_padded"] = qa_df["answer_padded"].apply(ast.literal_eval)

    print("Creating train/val/test splits...")
    # Recreate train/val/test split with same random state as notebook/evals
    X = np.array(qa_df["question_padded"].tolist())
    y = np.array(qa_df["answer_padded"].tolist())
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print("Data loaded successfully!")
    print(f"    - Vocabulary size: {len(word2index)}")
    print(f"    - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, word2index, index2word


def get_vocab_size(word2index):
    """Get vocabulary size"""
    return len(word2index)


def print_data_summary(X_train, y_train, X_val, y_val, X_test, y_test, word2index):
    """Print summary of loaded data"""
    print("\n" + "=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(f"Vocabulary size: {len(word2index)}")
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(
        f"Sequence length: {X_train.shape[1]} (questions), {y_train.shape[1]} (answers)"
    )
    print("Special tokens:", ["<pad>", "<unk>", "<sos>", "<eos>"])
    print("=" * 50 + "\n")


# TOKENIZATION AND VOCABULARY UTILITIES


def tokenize(text):
    """Tokenize input using spacy tokenizer"""
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


def build_vocab(pairs, min_freq=2):
    """Build vocabulary from pairs of (source, target) text"""
    counter = Counter()
    for source, target in pairs:
        counter.update(tokenize(source))
        counter.update(tokenize(target))

    # Build vocab
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + [
        token for token, freq in counter.items() if freq >= min_freq
    ]

    word2idx = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for w, i in word2idx.items()}
    return word2idx, index_to_word


def numericalize(text, word2idx):
    """Convert text to numerical sequence with special tokens"""
    return (
        [word2idx.get(SOS_TOKEN)]
        + [word2idx.get(tok, word2idx[UNK_TOKEN]) for tok in tokenize(text)]
        + [word2idx.get(EOS_TOKEN)]
    )


# DATASET AND DATA LOADING


class ChatDataset(Dataset):
    """Dataset class for chat data"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    """Collate function for DataLoader - zips source and target tokens into tensors"""
    srcs, tgts = zip(*batch)
    # Convert to numpy arrays first for better performance
    src_array = np.array(srcs)
    tgt_array = np.array(tgts)
    # Then create tensors from numpy arrays
    src_tensor = torch.from_numpy(src_array).long().transpose(0, 1)
    tgt_tensor = torch.from_numpy(tgt_array).long().transpose(0, 1)
    return src_tensor.to(device), tgt_tensor.to(device)


# MODEL ARCHITECTURES


class Encoder(nn.Module):
    """Encoder class - Processes sequence in both directions"""

    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, src):
        embedded = self.embedding(src)  # [src_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)  # pass into GRU
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
        ).unsqueeze(0)  # merge both direction hidden states
        return outputs, hidden


class LuongAttention(nn.Module):
    """Luong attention mechanism"""

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.squeeze(0)  # Remove time from hidden state
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = self.attn(encoder_outputs)
        scores = torch.bmm(energy, hidden.unsqueeze(2))
        attn_weights = F.softmax(scores, dim=1)  # get attention weights
        context = torch.bmm(
            attn_weights.transpose(1, 2), encoder_outputs
        )  # compute context vector
        return context.transpose(0, 1)  # [1, batch_size, 512]


class Decoder(nn.Module):
    """Decoder with Luong attention - Assumes BiRNN encoder"""

    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attn = LuongAttention(enc_hid_dim, dec_hid_dim)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim * 2, dec_hid_dim)
        self.fc = nn.Linear(dec_hid_dim + enc_hid_dim * 2, vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)  # Add time dimension
        embedded = self.embedding(input)  # [1, batch_size, 256]
        context = self.attn(hidden, encoder_outputs)  # compute context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)  # Update hidden states
        output = self.fc(
            torch.cat((output, context), dim=2).squeeze(0)
        )  # predict token
        return output, hidden


# TRAINING AND SAMPLING FUNCTIONS


def train(
    encoder,
    decoder,
    loader,
    optimizer,
    criterion,
    n_epochs=25,
    teacher_forcing_ratio=0.5,
):
    """Train the seq2seq model"""
    CLIP = 1.0
    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        teacher_forcing_ratio = max(0.5, 1.0 - epoch * 0.01)

        for src_batch, tgt_batch in loader:
            optimizer.zero_grad()
            enc_outputs, enc_hidden = encoder(src_batch)
            dec_input = tgt_batch[0, :]
            dec_hidden = enc_hidden
            loss = 0

            for t in range(1, tgt_batch.size(0)):
                dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_outputs)
                loss += criterion(dec_output, tgt_batch[t])
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = dec_output.argmax(1)
                dec_input = tgt_batch[t] if teacher_force else top1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item() / tgt_batch.size(0)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(loader):.4f}")


def top_k_sampling(logits, k=10):
    """Top-k sampling to reduce bias towards common tokens"""
    probability = F.softmax(logits, dim=-1)
    top_k_probability, top_k_index = torch.topk(probability, k)
    top_k_probability = top_k_probability / top_k_probability.sum()
    sampled_index = torch.multinomial(top_k_probability, 1)
    return top_k_index[sampled_index].item()


def chat(encoder, decoder, input_text, word_to_index, index_to_word, max_len=20):
    """Generate response using trained model"""
    encoder.eval()
    decoder.eval()

    # Tokenize and parse response
    tokens = tokenize(input_text)
    token_ids = (
        [word_to_index.get(SOS_TOKEN)]
        + [word_to_index.get(tok, word_to_index[UNK_TOKEN]) for tok in tokens]
        + [word_to_index.get(EOS_TOKEN)]
    )
    src_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)
    enc_outputs, enc_hidden = encoder(src_tensor)
    dec_input = torch.tensor([word_to_index[SOS_TOKEN]], device=device)
    dec_hidden = enc_hidden
    result = []

    for _ in range(max_len):
        dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_outputs)
        token = top_k_sampling(dec_output.squeeze(0), k=10)
        if token == word_to_index[EOS_TOKEN]:
            break
        word = index_to_word.get(token, UNK_TOKEN)
        result.append(word)
        dec_input = torch.tensor([token], device=device)

    return " ".join(result)


# MODEL CREATION HELPERS=


def create_model_components(
    vocab_size, enc_emb_dim=128, dec_emb_dim=128, enc_hid_dim=256, dec_hid_dim=256
):
    """Create encoder, decoder, optimizer and criterion"""
    encoder = Encoder(vocab_size, enc_emb_dim, enc_hid_dim).to(device)
    decoder = Decoder(vocab_size, dec_emb_dim, dec_hid_dim, enc_hid_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters())
    )
    return encoder, decoder, optimizer


def save_model(encoder, decoder, model_path):
    """Save trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")


def load_model(
    model_path,
    vocab_size,
    enc_emb_dim=128,
    dec_emb_dim=128,
    enc_hid_dim=256,
    dec_hid_dim=256,
):
    """Load trained model"""
    encoder = Encoder(vocab_size, enc_emb_dim, enc_hid_dim).to(device)
    decoder = Decoder(vocab_size, dec_emb_dim, dec_hid_dim, enc_hid_dim).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    return encoder, decoder
