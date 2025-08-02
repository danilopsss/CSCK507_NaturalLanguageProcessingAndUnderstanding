import os
import sys
import torch
import torch.nn as nn

# Add src directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.insert(0, src_dir)

# Import consolidated utilities
# Must happen after after path setup!
from utils.model_utils import (
    load_preprocessed_data,
    ChatDataset,
    collate_fn,
    create_model_components,
    train,
    chat,
    save_model,
    PAD_TOKEN,
)

# Load preprocessed data
X_train, y_train, X_val, y_val, X_test, y_test, word2idx, idx2word = (
    load_preprocessed_data()
)

# Create training data
pairs = list(zip(X_train, y_train))
train_data = ChatDataset(pairs)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=200, shuffle=True, collate_fn=collate_fn
)

# Model hyperparameters
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
vocab_size = len(word2idx)

# Create model components
encoder, decoder, optimizer = create_model_components(
    vocab_size, ENC_EMB_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM
)

# Create criterion
criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])

# Train the model
train(encoder, decoder, train_loader, optimizer, criterion)

# Save the model
project_root = os.path.dirname(os.path.dirname(script_dir))
model_path = os.path.join(project_root, "models", "chatbot_model_with_attention.pth")
save_model(encoder, decoder, model_path)

# Test Chat
while True:
    query = input("You: ")
    if query.lower() in ["quit", "exit"]:
        break
    response = chat(encoder, decoder, query, word2idx, idx2word)
    print(f"Bot: {response}")
