import torch

from pathlib import Path

from src.utils.encoder_no_attention import EncoderNoAttention
from src.utils.decoder_no_attention import DecoderNoAttention
from src.utils.chat import Vocab
from src.utils.device import device


NO_ATTENTION_WEIGHTS = next(
    Path(__file__).parent.parent.rglob("**/*/chatbot_model_no_attention.pth")
)
CHECKPOINT = torch.load(NO_ATTENTION_WEIGHTS, map_location=device)

EMBEDDING_DIM = 256
HIDDEN_SIZE = 512


def no_attention_encoder():
    vocab = Vocab()
    return (
        EncoderNoAttention(
            len(vocab),
            EMBEDDING_DIM,
            HIDDEN_SIZE,
        )
        .to(device)
        .eval()
    )


def no_attention_decoder():
    vocab = Vocab()
    return (
        DecoderNoAttention(
            len(vocab),
            EMBEDDING_DIM,
            HIDDEN_SIZE,
        )
        .to(device)
        .eval()
    )
