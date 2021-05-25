def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')
    group.add_argument('--dict', type=str, 
            help='Path to dictionary file')
    group.add_argument('--k', type=int, 
            help='window size')


class Tokenizer():
    def __init__(self, args):
        with open(args.dict, 'r') as f:
            words = f.read().splitlines()
            self.dict = {word: idx for idx, word in enumerate(words)}
            
            self.k = args.k
            
    def get_vocab_size(self):
        return len(self.dict) + 1

    def tokenize(self, sentence,k=1):
        tokens = [0]*(k-1)
        for word in sentence.split():
            tokens.append(self.dict.get(word, len(self.dict)))
        return tokens

