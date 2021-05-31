import re
import numpy as np


def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')


class Tokenizer():
    def __init__(self, args):
        self.dict = {}

    def get_vocab_size(self):
        return ord('z') - ord('a') + 1

    def tokenize(self, sentence):
        regex = re.compile('[^a-zA-Z]')
        sentence = regex.sub('', sentence)
        sentence = sentence.lower()

        tokens = []
        for ch in sentence:
            tokens.append(ord(ch) - ord('a') + 1)

        tokens = np.array(tokens)
        return tokens

