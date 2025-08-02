import torch
import pickle
from Model_Utils import Encoder, Decoder, tokenize, top_k_sampling  # Replace with your actual model file and classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#open stored wordToIndex and IndextToWord vectors
with open('D:\\Final_Project\\CSCK507_NaturalLanguageProcessingAndUnderstanding\\Data processing\\word2index.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('D:\\Final_Project\\CSCK507_NaturalLanguageProcessingAndUnderstanding\\Data processing\\index2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)
    
# Model hyperparameters must match what was used during training
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
vocab_size = 5748

# Initialize models
encoder = Encoder(vocab_size, ENC_EMB_DIM, ENC_HID_DIM).to(device)
decoder = Decoder(vocab_size, DEC_EMB_DIM, DEC_HID_DIM, ENC_HID_DIM).to(device)

# Load weights
checkpoint = torch.load('D:\Final_Project\CSCK507_NaturalLanguageProcessingAndUnderstanding\Model\chatbot_model.pth', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

def chat(encoder, decoder, input_text, word2idx, idx2word, max_len=20):
    encoder.eval()
    decoder.eval()
    
    #tokenize and parse response
    #token must be formatted with correct <sos> and <eos> tags
    tokens = tokenize(input_text)
    token_ids = [word2idx.get(SOS_TOKEN)] + \
                [word2idx.get(tok, word2idx[UNK_TOKEN]) for tok in tokens] + \
                [word2idx.get(EOS_TOKEN)]
    src_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)
    enc_outputs, enc_hidden = encoder(src_tensor)
    dec_input = torch.tensor([word2idx[SOS_TOKEN]], device=device)
    dec_hidden = enc_hidden
    result = []
    for _ in range(max_len):
        dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_outputs)
        #call top 10 k sampling to remove bias towards common tokens
        token = top_k_sampling(dec_output.squeeze(0), k=10)
        if token == word2idx[EOS_TOKEN]:
            break
        word = idx2word.get(token, UNK_TOKEN)
        result.append(word)
        dec_input = torch.tensor([token], device=device)
    return ' '.join(result) #seperate response by space

#Chat here
while True:
    query = input("You: ")
    if query.lower() in ['quit', 'exit']:
        break
    response = chat(encoder, decoder, query, word2idx, idx2word) 
    print(f"Bot: {response}")
