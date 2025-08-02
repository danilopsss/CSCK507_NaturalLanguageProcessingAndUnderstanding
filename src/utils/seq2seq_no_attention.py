import random
from torch import nn, zeros


class Seq2SeqNoAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.0):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        outputs = zeros(batch_size, trg_len, vocab_size).to(self.device)

        hidden = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs