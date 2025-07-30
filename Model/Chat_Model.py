import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import spacy
from collections import Counter
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import ast

# Use spaCy tokenization like in preprocessing class
spacy_en = spacy.load('en_core_web_lg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters must match what was used during training
SOS_TOKEN = "<sos>"
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Tokenization and vocabulary
def tokenize(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)] #tokenize input using spacy tokenizer
def build_vocab(pairs, min_freq=2):
    counter = Counter()
    for source, target in pairs:
        counter.update(tokenize(source))
        counter.update(tokenize(target))
    #build vocab
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + \
            [token for token, freq in counter.items() if freq >= min_freq]

    word_to_index = {w: i for i, w in enumerate(vocab)} #map each word to a unique integer index
    index_to_word = {i: w for w, i in word2idx.items()} #also create reverse mapping
    return word_to_index, index_to_word
#function to add start of sentence token, mark unknown tokens, and add end of sentence token
def numericalize(text, word2idx):
    return [word2idx.get(SOS_TOKEN)] + \
           [word2idx.get(tok, word2idx[UNK_TOKEN]) for tok in tokenize(text)] + \
           [word2idx.get(EOS_TOKEN)]
# Dataset
class ChatDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
#collate zips source and target tokens from padded data into seq_len, batch_size tensors.
def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_tensor = torch.tensor(srcs, dtype=torch.long).transpose(0, 1)
    tgt_tensor = torch.tensor(tgts, dtype=torch.long).transpose(0, 1)
    return src_tensor.to(device), tgt_tensor.to(device)
#Encoder class - Processes sequence in both directions
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim) #5748 x 256
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim) #linear layer combines both directions -> 512 
        self.dropout = nn.Dropout(0.3) #Reduce overfitting

    def forward(self, src):
        embedded = self.embedding(src) #[src_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded) #pass into GRU
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))).unsqueeze(0) #merge both direction hidden states
        return outputs, hidden
class LuongAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2, dec_hid_dim) # maps encoder output to same size as decoder's hidden size
        self.dropout = nn.Dropout(0.3) #Reduce overfitting

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.squeeze(0) #Remove time from hidden state
        encoder_outputs = encoder_outputs.permute(1, 0, 2) #small optimization for faster batching
        energy = self.attn(encoder_outputs)
        scores = torch.bmm(energy, hidden.unsqueeze(2)) 
        attn_weights = F.softmax(scores, dim=1) #get attention weights
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs) #compute context vector
        return context.transpose(0, 1) #[1, batch_size, 512]
#Decoder - Assumes BiRNN encoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim) #embedd input
        self.attn = LuongAttention(enc_hid_dim, dec_hid_dim) #LUONG Attention 
        self.rnn = nn.GRU(emb_dim + enc_hid_dim * 2, dec_hid_dim) #GRU update decoder hidden state
        self.fc = nn.Linear(dec_hid_dim + enc_hid_dim * 2, vocab_size) #map decoder output to vocab size

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0) #Add time dimension for decoder to process sequentially
        embedded = self.embedding(input) #[1, batch_size, 256]
        context = self.attn(hidden, encoder_outputs) #compute context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden) #Update hidden states
        output = self.fc(torch.cat((output, context), dim=2).squeeze(0)) #predict token
        return output, hidden
def train(encoder, decoder, loader, optimizer, criterion, n_epochs=200, teacher_forcing_ratio=0.5):
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
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")            
def top_k_sampling(logits, k=10):
    probability = F.softmax(logits, dim=-1) #convert logits to prob
    top_k_probability, top_k_index = torch.topk(probability, k) #select top 10 most probably tokens
    top_k_probability = top_k_probability / top_k_probability.sum()  # normalize values to 1
    sampled_index = torch.multinomial(top_k_probability, 1) #grab a random sample of 1 of the 10 values
    return top_k_index[sampled_index].item()
def chat(encoder, decoder, input_text, word_to_index, index_to_word, max_len=20):
    encoder.eval()
    decoder.eval()
    
    #tokenize and parse response
    #token must be formatted with correct <sos> and <eos> tags
    tokens = tokenize(input_text)
    token_ids = [word_to_index.get(SOS_TOKEN)] + \
                [word_to_index.get(tok, word_to_index[UNK_TOKEN]) for tok in tokens] + \
                [word_to_index.get(EOS_TOKEN)]
    src_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)
    enc_outputs, enc_hidden = encoder(src_tensor)
    dec_input = torch.tensor([word_to_index[SOS_TOKEN]], device=device)
    dec_hidden = enc_hidden
    result = []
    for _ in range(max_len):
        dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_outputs)
        #call top 10 k sampling to remove bias towards common tokens
        token = top_k_sampling(dec_output.squeeze(0), k=10)
        if token == word2idx[EOS_TOKEN]:
            break
        word = index_to_word.get(token, UNK_TOKEN)
        result.append(word)
        dec_input = torch.tensor([token], device=device)
    return ' '.join(result) #seperate response by space

#open stored wordToIndex and IndextToWord vectors
with open('D:\\Final_Project\\CSCK507_NaturalLanguageProcessingAndUnderstanding\\Data processing\\word2index.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('D:\\Final_Project\\CSCK507_NaturalLanguageProcessingAndUnderstanding\\Data processing\\index2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)
qa_df = pd.read_csv('D:\\Final_Project\\CSCK507_NaturalLanguageProcessingAndUnderstanding\\Data processing\\DATASET_PROCESSED.csv')

#grab padded answers from dataset. Convert to literals for list-zip
qa_df['question_padded'] = qa_df['question_padded'].apply(ast.literal_eval)
qa_df['answer_padded'] = qa_df['answer_padded'].apply(ast.literal_eval)

#Perform train/test split - This is done locally since qa_df was saved from notebook.
#TODO: pickle X and Y from ipynb preprocessing
X = np.array(qa_df['question_padded'].to_numpy())
y = np.array(qa_df['answer_padded'].to_numpy())
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
pairs = list(zip(X_train, y_train))
data = pairs

#Use dataloader to batch training data
train_data = ChatDataset(data)
train_loader = DataLoader(train_data, batch_size=200, shuffle=True, collate_fn=collate_fn)

ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
vocab_size = len(word2idx) #5748

#Create encoder, decoder, Adam optimizer and use CEL criterion.
#ignore padded tokens on loss
encoder = Encoder(vocab_size, ENC_EMB_DIM, ENC_HID_DIM).to(device)
decoder = Decoder(vocab_size, DEC_EMB_DIM, DEC_HID_DIM, ENC_HID_DIM).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])

#train
train(encoder, decoder, train_loader, optimizer, criterion)

# Save the model
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
}, 'Model/chatbot_model.pth')

# Test Chat
while True:
    query = input("You: ")
    if query.lower() in ['quit', 'exit']:
        break
    response = chat(encoder, decoder, query, word2idx, idx2word)
    print(f"Bot: {response}")
