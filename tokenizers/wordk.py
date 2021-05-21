  def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')
    group.add_argument('--dict', type=str, 
            help='Path to dictionary file')


class Tokenizer():
    def __init__(self, args):
        with open(args.dict, 'r') as f:
            words = f.read().splitlines()
            self.dict = {word: idx for idx, word in enumerate(words)}

    def get_vocab_size(self):
        return len(self.dict) + 1

    def tokenize(self, sentence,k=15):
        tokens = [len(self.dict)+1]*(k-1) # token padding
        for word in sentence.split():
            tokens.append(self.dict.get(word, len(self.dict)))
        return tokens
