import torch

from pathlib import Path

from src.models.encoders.encoder_no_attention import EncoderNoAttention
from src.models.decoders.decoder_no_attention import DecoderNoAttention
from src.models.seq2seq.seq2seq_no_attention import Seq2SeqNoAttention
from src.chatbot.chat import Vocab
from src.utils.device import device


NO_ATTENTION_WEIGHTS = Path(__file__).parent.parent.rglob(
    "**/*/chatbot_model_no_attention_2.pth"
)
CHECKPOINT = torch.load(next(NO_ATTENTION_WEIGHTS), map_location=device)

EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
VOCAB = Vocab()


def no_attention_encoder():
    return (
        EncoderNoAttention(
            len(VOCAB),
            EMBEDDING_DIM,
            HIDDEN_SIZE,
        )
        .to(device)
    )


def no_attention_decoder():
    return (
        DecoderNoAttention(
            len(VOCAB),
            EMBEDDING_DIM,
            HIDDEN_SIZE,
        )
        .to(device)
    )


def natt_model():
    model = Seq2SeqNoAttention(
        encoder=no_attention_encoder(),
        decoder=no_attention_decoder(),
        device=device
    ).to(device)
    model.load_state_dict(CHECKPOINT)
    model.eval()
    return model
