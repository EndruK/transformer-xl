from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.cuda

from mem_transformer import MemTransformerLM
from utils.vocabulary import Vocab


class ModelWrapper:
    def __init__(self, model: MemTransformerLM,
                 vocab: Vocab,
                 sp_processor: spm.SentencePieceProcessor,
                 device: str):
        self.vocab = vocab
        self.sp_processor = sp_processor
        self.device = device
        self.model = model.to(device=self.device)
        self.model.eval()

    @classmethod
    def load(cls, model_path: Path, spm_path: Path,
             device: str = None) -> 'ModelWrapper':
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with model_path.open('rb') as f:
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
        sp_processor = spm.SentencePieceProcessor()
        sp_processor.Load(str(spm_path))
        return cls(model, vocab, sp_processor, device)

    def predict(self, text: str):
        tokens = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            tokens.extend(self.sp_processor.encode_as_pieces(line))
            assert not self.vocab.add_double_eos
            if self.vocab.add_eos and i != len(lines) - 1:
                tokens.append(self.vocab.EOS)
        all_xs = self.vocab.convert_to_tensor(tokens)
        all_log_probs = []
        with torch.no_grad():
            mems = tuple()
            batch_size = self.model.tgt_len
            for idx in range(0, len(all_xs), batch_size):
                xs = all_xs[idx: idx + batch_size]
                xs = xs.to(device=self.device)
                target = None
                log_probs, mems = self.model(xs.unsqueeze(0), target, *mems)
                log_probs = log_probs.squeeze(0).data.cpu().numpy()
                all_log_probs.append(log_probs)
        return np.concatenate(all_log_probs)
