import re
from transformers import AutoTokenizer


def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')
    group.add_argument('--tokenizer_name', type=str, 
            help='Name of pretrained tokenizer')


class Tokenizer():
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def tokenize(self, sentence):
        return self.tokenizer(sentence, add_special_tokens=True).get('input_ids')

