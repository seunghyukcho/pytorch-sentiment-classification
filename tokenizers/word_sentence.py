from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
import nltk
nltk.download('punkt')


def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')
    group.add_argument('--dict', type=str, 
            help='Path to word dictionary file')
    group.add_argument('--sentence_dict',type=str,
            help='Path to sentence dictionary file')


class Tokenizer():
    def __init__(self, args):
        with open(args.dict, 'r') as f:
            words = f.read().splitlines()
            self.word_dict = {word: idx for idx, word in enumerate(words)}
        
        with open(args.sentence_dict, 'r') as f:
            sentences = f.read().splitlines()
            self.sentence_dict = {s: idx for idx, s in enumerate(sentences)}            
            
        self.word_tokenizer = TreebankWordTokenizer()
        
    def get_vocab_size(self):
        return len(self.word_dict) + 1
    
    def get_sentence_size(self):
        return len(self.sentence_dict) + 1 
    
    def tokenize(self, sentence):
       
        sentence_tokens = list(map(lambda x:self.sentence_dict[x] if x in self.sentence_dict else len(self.sentence_dict) , sent_tokenize(sentence)))
        word_tokens = list(map(lambda x:self.word_dict[x] if x in self.word_dict else len(self.word_dict), self.word_tokenizer.tokenize(sentence)))
        
        return word_tokens, sentence_tokens
