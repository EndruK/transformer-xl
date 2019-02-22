from pathlib import Path

import torch

from mem_transformer import MemTransformerLM
from utils.vocabulary import Vocab


# TODO load sentencepiece model


class ModelWrapper:
    def __init__(self, model: MemTransformerLM, vocab: Vocab):
        self.vocab = vocab
        self.model = model

    @classmethod
    def load(cls, path: Path) -> 'ModelWrapper':
        with path.open('rb') as f:
            state = torch.load(f, map_location='cpu')
        model = MemTransformerLM(**state['model_params'])
        model.load_state_dict(state['state_dict'])
        vocab_params = state['vocab_params']
        vocab = Vocab.from_symbols(
            state['vocab'],
            lower_case=vocab_params['lower_case'],
            add_eos=vocab_params['add_eos'],
            add_double_eos=vocab_params['add_double_eos'],
        )
        return cls(model, vocab)

