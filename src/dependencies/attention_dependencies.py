
import torch
from pathlib import Path
from src.models.encoders.encoder_attention import EncoderWithAttention
from src.models.decoders.decoder_attention import DecoderWithAttention
from src.models.seq2seq.seq2seq_attention import Seq2SeqWithAttention
from src.chatbot.chat import Vocab
from src.utils.device import device

ENC_EMB_DIM = 128
DEC_EMB_DIM = ENC_EMB_DIM
ENC_HID_DIM = 256
DEC_HID_DIM = ENC_HID_DIM

ATTENTION_WEIGHTS = next(Path(__file__).parent.parent.rglob(
    "**/*/chatbot_model_with_attention.pth")
)

CHECKPOINT = torch.load(ATTENTION_WEIGHTS, map_location=device)
VOCAB = Vocab()


def attention_encoder():
    encoder = (
        EncoderWithAttention(
            len(VOCAB),
            ENC_EMB_DIM,
            ENC_HID_DIM
        )
        .to(device)
    )
    encoder.load_state_dict(CHECKPOINT.get("encoder_state_dict"))
    encoder.eval()
    return encoder


def attention_decoder():
    decoder = (
        DecoderWithAttention(
            len(VOCAB),
            DEC_EMB_DIM,
            ENC_HID_DIM,
            DEC_HID_DIM
        )
        .to(device)
    )
    decoder.load_state_dict(CHECKPOINT.get("decoder_state_dict"))
    decoder.eval()
    return decoder


def att_model():
    encoder = attention_encoder()
    decoder = attention_decoder()
    return Seq2SeqWithAttention(
        encoder=encoder,
        decoder=decoder,
    )
