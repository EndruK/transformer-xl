from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
import torch
import torch.cuda

from mem_transformer import MemTransformerLM
from utils.vocabulary import Vocab


class ModelWrapper:
    def __init__(self, model: MemTransformerLM,
                 vocab: Vocab,
                 device: str,
                 batch_size: int):
        self.vocab = vocab
        self.sp_processor = sp_processor
        self.device = device
        self.batch_size = batch_size
        self.model = model.to(device=self.device)
        self.model.eval()

    @classmethod
    def load(cls, model_path: Path,
             batch_size: int,
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
            #add_eos=vocab_params['add_eos'],
            #add_double_eos=vocab_params['add_double_eos'],
        )
        return cls(model, vocab, device, batch_size)

    def tokenize(self, text: str) -> List[str]:
        #tokens = self.vocab.tokenize(text)
        tokens = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            line_tokens = self.vocab.tokenize(line)
            tokens.extend(line_tokens)

        # tokens = []
        # lines = text.split('\n')
        # for i, line in enumerate(lines):
        #     line = line.strip()
        #     foo = self.sp_processor.encode_as_pieces(line)
        #     tokens.extend(foo)
        #     assert not self.vocab.add_double_eos
        #     if self.vocab.add_eos and i != len(lines) - 1:
        #         tokens.append(self.vocab.EOS)
        return tokens

    def get_log_probs(self, tokens: List[str]) -> torch.Tensor:
        """ Return log probabilities for next tokens.
        Shape of returned tensor is len(tokens) x len(self.vocab),
        where the first element contains log probabilities for tokens
        after the first, and last element log probabilities for tokens
        after the last one.
        """
        if not tokens:
            raise ValueError('tokens must be non-empty')
        all_xs = self.vocab.convert_to_tensor(tokens)
        all_log_probs = []
        with torch.no_grad():
            mems = tuple()
            input_len = self.model.tgt_len
            for idx in range(0, len(all_xs), input_len):
                xs = all_xs[idx: idx + input_len]
                xs = xs.to(device=self.device)
                batch_dim = 1  # batch size dimension is 1
                xs = xs.unsqueeze(batch_dim)
                log_probs, mems = self.model(xs, None, *mems)
                log_probs = log_probs.squeeze(batch_dim).data.cpu()
                all_log_probs.append(log_probs)
        return torch.cat(all_log_probs)

    def next_top_k(
            self, tokens: List[str], top_k: int = 40,
            ) -> List[Tuple[str, float]]:
        """ Return top k next tokens and their log probabilities.
        """
        log_probs = self.get_log_probs(tokens)[-1]
        top_indices = torch.argsort(log_probs)[-top_k:]
        top_log_probs = log_probs[top_indices]
        return [(self.vocab.idx2sym[idx], log_prob.item())
                for idx, log_prob in
                reversed(list(zip(top_indices, top_log_probs)))]

    def sample_next(self, tokens: List[str], top_k: int = 40) -> str:
        """ Sample next token from multinomial distribution.
        """
        log_probs = self.get_log_probs(tokens)[-1]
        top_indices = torch.argsort(log_probs)[-top_k:]
        top_probs = log_probs[top_indices].double().exp()
        sampled_idx = top_indices[torch.multinomial(top_probs, 1).item()].item()
        return self.vocab.idx2sym[sampled_idx]

    def sample_text_iter(self, text: str, top_k: int = 40):
        """ An iterator yielding pieces of generated text, resulting text
        can be obtained by joining all of them with an empty string.
        """
        # TODO for longer texts we want to use memory and don't feed all tokens
        tokens = self.tokenize(text)
        while True:
            next_token = self.sample_next(tokens, top_k=top_k)
            yield (self.sp_processor.DecodePieces([tokens[-1], next_token])
                   [len(self.sp_processor.DecodePieces([tokens[-1]])):])
            tokens.append(next_token)
