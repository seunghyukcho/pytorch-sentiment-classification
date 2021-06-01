  
from nltk.tokenize import sent_tokenize
import re

def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')
    group.add_argument('--dict', type=str, 
            help='Path to word dictionary file')
    group.add_argument('--k', type=int, 
            help='window size')
            
class Tokenizer():
    def __init__(self, args):
        with open(args.dict, 'r') as f:
            words = f.read().splitlines()
            self.word_dict = {word: idx for idx, word in enumerate(words)}       
        self.k = args.k

    def get_vocab_size(self):
        return len(self.word_dict) + 1
    
    def tokenize(self, sentence):
        sentence = re.sub("and","",sentence)
        word_tokens = [0 for _ in range(self.k-1)]
        sent_token = sent_tokenize(sentence)      
        word_tokens = word_tokens + list(map(lambda x:self.word_dict[x] if x in self.word_dict else len(self.word_dict), sent_tokenize(sentence)))
        
        return word_tokens
        
