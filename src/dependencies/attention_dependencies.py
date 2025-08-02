
import torch
from pathlib import Path
from src.utils.encoder_attention import EncoderWithAttention
from src.utils.decoder_attention import DecoderWithAttention
from src.utils.seq2seq_no_attention import Seq2SeqNoAttention
from src.utils.chat import Vocab
from src.utils.device import device

ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256


ATTENTION_WEIGHTS = next(Path(__file__).parent.parent.rglob("**/*/chatbot_model_with_attention.pth"))

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
    encoder.load_state_dict(CHECKPOINT["encoder_state_dict"])
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
    decoder.load_state_dict(CHECKPOINT["decoder_state_dict"])
    decoder.eval()
    return decoder


def no_attention_model():
    encoder = attention_encoder()
    decoder = attention_decoder()
    model = Seq2SeqNoAttention(
        encoder=encoder,
        decoder=decoder
    ).to(device)
    return model.load_state_dict(
        torch.load(ATTENTION_WEIGHTS, map_location=device)
    )
