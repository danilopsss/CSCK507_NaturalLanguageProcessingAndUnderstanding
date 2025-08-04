import spacy
from nltk.tokenize import TreebankWordTokenizer


class DefaultTokenizers:
    def __init__(self):
        self._spacy_dialect = spacy.load("en_core_web_lg")
        self._tree_bank_tokenizer = TreebankWordTokenizer()

    def spacy_tokenizer(self, text: str):
        """Tokenize input using spacy tokenizer"""
        return [
            tok.text.lower() for tok
            in self._spacy_dialect.tokenizer(text)
        ]

    def tree_bank_tokenizer(self, text: str):
        return [
            tok.lower() for tok
            in self._tree_bank_tokenizer.tokenize(text)
        ]
