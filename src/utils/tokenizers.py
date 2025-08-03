from spacy.language import Language


class DefaultTokenizers:
    def __init__(self, spacy_dialect: Language):
        self._spacy_dialect = spacy_dialect

    def default_tokenizer(self, text: str):
        """Tokenize input using spacy tokenizer"""
        return [tok.text.lower() for tok in self._spacy_dialect.tokenizer(text)]
