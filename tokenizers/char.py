import re
import numpy as np


class Tokenizer():
    def __init__(self):
        self.dict = {}

    def tokenize(self, sentence):
        regex = re.compile('[^a-zA-Z]')
        sentence = regex.sub('', sentence)
        sentence = sentence.lower()

        tokens = []
        for ch in sentence:
            tokens.append(ord(ch) - ord('a'))

        tokens = np.array(tokens)
        tokens = np.eye(ord('z') - ord('a') + 1)[tokens]

        return tokens

