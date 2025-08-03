from torch import nn, bmm


class LuongAttention(nn.Module):
    """Luong attention mechanism"""

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.squeeze(0)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = self.attn(encoder_outputs)
        scores = bmm(energy, hidden.unsqueeze(2))
        attn_weights = nn.functional.softmax(scores, dim=1)
        context = bmm(
            attn_weights.transpose(1, 2), encoder_outputs
        )
        return context.transpose(0, 1)
