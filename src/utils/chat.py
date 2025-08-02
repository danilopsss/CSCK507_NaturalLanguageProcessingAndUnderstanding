import torch
import pickle
from pathlib import Path
from .hyperparameters import SOS_TOKEN, UNK_TOKEN, EOS_TOKEN
from .encoder_attention import EncoderWithAttention
from .encoder_no_attention import EncoderNoAttention
from .decoder_attention import DecoderWithAttention
from .decoder_no_attention import DecoderNoAttention


class Vocab:
    def __init__(self):
        self._vocab_dict = None

    @property
    def vocab_dict(self) -> dict:
        if self._vocab_dict is None:
            vocab_path = next(Path(".").parent.rglob("**/*/word2index.pkl"))
            with open(vocab_path, mode="rb") as f:
                self._vocab_dict = pickle.load(f)
        return self._vocab_dict

    def __len__(self):
        return len(self.vocab_dict)

    def index_to_word(self, idx: int) -> str:
        return {v: k for k, v in self.vocab_dict.items()}.get(idx, UNK_TOKEN)


class Chat:
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256

    def __init__(
            self,
            encoder:  EncoderWithAttention | EncoderNoAttention,
            decoder:  DecoderWithAttention | DecoderNoAttention,
            device: torch.DeviceObjType,
            tokenizer: callable
        ):
        self.vocab = Vocab()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.tokenizer = tokenizer

    def token_to_id(self, tokenized: list) -> list:
        indexed_tokens = [
            self.vocab.vocab_dict.get(tok, self.vocab.vocab_dict[UNK_TOKEN])
            for tok in tokenized
        ]
        return [
            self.vocab.vocab_dict.get(SOS_TOKEN),
            *indexed_tokens,
            self.vocab.vocab_dict.get(EOS_TOKEN)
        ]

    def top_k_sampling(self, logits, k=10) -> torch.NumberType:
        """Top-k sampling to reduce bias towards common tokens"""
        probability = torch.nn.functional.softmax(logits, dim=-1)
        top_k_probability, top_k_index = torch.topk(probability, k)
        top_k_probability = top_k_probability / top_k_probability.sum()
        sampled_index = torch.multinomial(top_k_probability, 1)
        return top_k_index[sampled_index].item()

    def process_encoder_output(self, output: tuple | torch.TensorType, max_len: int = 20) -> list:
        breakpoint()
        if len(output) == 1:
            ...
        else:
            enc_outputs, enc_hidden = output
            dec_input = torch.tensor([
                self.vocab.vocab_dict.get(SOS_TOKEN)],
                device=self.device
            )
            dec_hidden = enc_hidden
            result = []

            for _ in range(max_len):
                dec_output, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs)
                token = self.top_k_sampling(dec_output.squeeze(0))
                if token == self.vocab.vocab_dict.get(EOS_TOKEN):
                    break
                word = self.vocab.index_to_word(token)
                result.append(word)
                dec_input = torch.tensor([token], device=self.device)
        return result

    def ask(self, user_input: str, max_len: int = 20) -> str:
        tokenized = self.tokenizer(user_input)
        tokens_ids = self.token_to_id(tokenized=tokenized)

        src_tensor = torch.tensor(
            tokens_ids, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        output = self.encoder(src_tensor)
        result = self.process_encoder_output(output)
        return " ".join(result)
