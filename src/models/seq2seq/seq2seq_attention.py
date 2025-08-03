

class Seq2SeqWithAttention:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
