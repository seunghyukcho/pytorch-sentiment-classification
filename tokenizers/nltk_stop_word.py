from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

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
        self.stop_words = set(stopwords.words('english'))

    def get_vocab_size(self):
        return len(self.word_dict) + 1
    
    def tokenize(self, sentence):
        sentence = sentence.replace(sentence[0], sentence[0].lower())
        chunk_sentence = word_tokenize(sentence)
        result = []
        for word in chunk_sentence: 
            if word not in self.stop_words: # 불용어 제거
                if len(word) > 1: # 단어 길이가 1 이하인 경우 제거
                    result.append(word)
                elif word == '!':
                    result.append(word)
                elif word == '?':
                    result.append(word)

        word_tokens = [0 for _ in range(self.k-1)]
        word_tokens = word_tokens + list(map(lambda x:self.word_dict[x] if x in self.word_dict else len(self.word_dict), result))
        
        return word_tokens
        
