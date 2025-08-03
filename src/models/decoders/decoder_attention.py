from torch import nn, cat
from .luiong_attention import LuongAttention


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attn = LuongAttention(enc_hid_dim, dec_hid_dim)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim * 2, dec_hid_dim)
        self.fc = nn.Linear(dec_hid_dim + enc_hid_dim * 2, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        input_token = input_token.unsqueeze(0)
        embedded = self.embedding(input_token)
        context = self.attn(hidden, encoder_outputs)
        rnn_input = cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(cat((output, context), dim=2).squeeze(0))
        return output, hidden
