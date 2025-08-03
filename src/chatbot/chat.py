import torch
import pickle
from pathlib import Path
from ..utils.hyperparameters import SOS_TOKEN, UNK_TOKEN, EOS_TOKEN, PAD_TOKEN
from src.models.seq2seq.seq2seq_no_attention import Seq2SeqNoAttention


class Vocab:
    def __init__(self):
        self._word_to_index = None
        self._index_to_word = None

    @property
    def word_to_index(self) -> dict:
        if self._word_to_index is None:
            vocab_path = next(Path(".").parent.rglob("**/*/word2index.pkl"))
            with open(vocab_path, mode="rb") as f:
                self._word_to_index = pickle.load(f)
        return self._word_to_index

    @property
    def index_to_word(self) -> dict:
        if self._index_to_word is None:
            vocab_path = next(Path(".").parent.rglob("**/*/index2word.pkl"))
            with open(vocab_path, mode="rb") as f:
                self._index_to_word = pickle.load(f)
        return self._index_to_word

    def __len__(self):
        return len(self.word_to_index)


class Chat:
    def __init__(
            self,
            model: Seq2SeqNoAttention,
            device: torch.DeviceObjType,
            tokenizer: callable
        ):
        self.vocab = Vocab()
        self.word2idx = self.vocab.word_to_index
        self.idx2word = self.vocab.index_to_word
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def ask(self, user_input: str):
        input_tokens = self.tokenizer(user_input)
        id_tokens = [
            self.word2idx.get(token, self.word2idx.get(UNK_TOKEN))
            for token in input_tokens
        ]
        sentence = [
            self.word2idx.get(SOS_TOKEN),
            *id_tokens,
            self.word2idx.get(EOS_TOKEN),
        ]

        input_tensor = torch.tensor(sentence, dtype=torch.long, device=self.device)
        input_tensor = input_tensor.unsqueeze(0)
        _, hidden = self.model.encoder(input_tensor)

        input_token = torch.tensor([self.word2idx.get(SOS_TOKEN)], device=self.device)

        with torch.no_grad():
            prediction = []

            for _ in range(20):
                output, hidden = self.model.decoder(input_token, hidden)
                next_token = output.argmax(1).item()

                if next_token == self.word2idx[EOS_TOKEN]:
                    break

                word = self.idx2word.get(next_token, UNK_TOKEN)

                if word not in [PAD_TOKEN, SOS_TOKEN]:
                    prediction.append(word)

                input_token = torch.tensor([next_token], device=self.device)
        return " ".join(prediction)
