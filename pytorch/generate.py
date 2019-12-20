from inference import ModelWrapper
from pathlib import Path
from typing import Tuple, List
import torch


class Generate:
    def __init__(self, model_path: str, generation_length: int, sample_size: int):
        self.model_path = model_path
<<<<<<< HEAD
        self.sample_size = sample_size
        self.mw = ModelWrapper.load(Path(self.model_path), batch_size=sample_size)
        self.length = generation_length
=======
        self.mw = ModelWrapper.load(Path(self.model_path), None)
>>>>>>> parent of 706f6fa... removed sentencepiece from inference

    def sample_next(self, tokens: List[str], top_k: int = 40) -> Tuple[str, float]:
        log_probs = self.mw.get_log_probs(tokens)[-1]
        top_indices = torch.argsort(log_probs)[-top_k:]
        top_probs = log_probs[top_indices].double().exp()
        multinomial = torch.multinomial(top_probs, 1).item()
        sampled_idx = top_indices[multinomial].item()
        return self.mw.vocab.idx2sym[sampled_idx], top_probs[multinomial].tolist()

    def generate(self, context: str, top_k: int = 40) -> List[Tuple[str, float]]:
        tokens = self.mw.tokenize(context)
        result = []
        for i in range(self.length):
            next_token, prob = self.sample_next(tokens, top_k=top_k)
            tokens.append(next_token)
            result.append((next_token, prob))
        return result


def main():
    path = "/home/andre/Documents/work/trained_models/transformer-xl-lopuhin/java-generic_dataset/20191216-160714/model.pt"
    string = "public static void main(String[] args) {\nSystem.out.println(\"Hello World!\");\n}"
<<<<<<< HEAD
    gen = Generate(model_path=model_path, generation_length=100)
=======
    gen = Generate(model_path=path)
>>>>>>> parent of 706f6fa... removed sentencepiece from inference
    result = gen.generate(context=string)
    print(result)


if __name__ == "__main__":
    main()
