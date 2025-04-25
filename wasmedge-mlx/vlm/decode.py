from transformers import AutoProcessor
import mlx.core as mx
import sys


def _remove_space(x):
    if x and x[0] == " ":
        return x[1:]
    return x


class Detokenizer():
    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value
        for i in range(len(self.tokenmap)):
            if self.tokenmap[i].startswith("<0x"):
                self.tokenmap[i] = chr(int(self.tokenmap[i][3:5], 16))

        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        v = self.tokenmap[token]
        if v[0] == "\u2581":
            if self.text or not self.trim_space:
                self.text += self._unflushed.replace("\u2581", " ")
            else:
                self.text = _remove_space(
                    self._unflushed.replace("\u2581", " "))
            self._unflushed = v
        else:
            self._unflushed += v


def decode(token: list, model_path: str, **kwargs):
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    detokenizer = Detokenizer(processor.tokenizer)
    for (i, token) in enumerate(token):
        detokenizer.add_token(token)
    return detokenizer.text


if __name__ == "__main__":
    model_path, output = sys.argv[1:]
    tokenList = mx.load(output)
    print(decode(tokenList.tolist(), model_path))
