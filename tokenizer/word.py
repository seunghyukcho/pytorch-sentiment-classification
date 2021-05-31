def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')
    group.add_argument('--dict', type=str, 
            help='Path to dictionary file')


class Tokenizer():
    def __init__(self, args):
        with open(args.dict, 'r') as f:
            words = f.read().splitlines()
            self.dict = {word: idx + 1 for idx, word in enumerate(words)}

    def get_pad_token_id(self):
        return 0

    def get_vocab_size(self):
        return len(self.dict) + 1

    def tokenize(self, sentence):
        tokens = []
        for word in sentence.split():
            tokens.append(self.dict.get(word, len(self.dict) + 1))
        return tokens

