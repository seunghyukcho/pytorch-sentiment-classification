from nltk.tokenize import TreebankWordTokenizer

def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')
    group.add_argument('--dict', type=str, 
            help='Path to word dictionary file')

class Tokenizer():
    def __init__(self, args):
        with open(args.dict, 'r') as f:
            words = f.read().splitlines()
            self.word_dict = {word: idx for idx, word in enumerate(words)}       
            
        self.word_tokenizer = TreebankWordTokenizer()
        
    def get_vocab_size(self):
        return len(self.dict) + 1
    
    def tokenize(self, sentence):
 
        word_tokens = list(map(lambda x:self.word_dict[x] if x in self.word_dict else len(self.word_dict), self.word_tokenizer(sentence)))
        
        return word_tokens
