import re
import numpy as np


def add_tokenizer_args(parser):
    group = parser.add_argument_group('tokenizer')


class Tokenizer():
    def __init__(self, args):
        self.dict = {}

    #전체 알파벳 갯수
    def get_vocab_size(self):
        return ord('z') - ord('a') + 1 #ord: 아스키 코드로 변환

    #
    def tokenize(self, sentence):
        regex = re.compile('[^a-zA-Z]') #문자가 아닌 것과 매치
        sentence = regex.sub('', sentence) #문자가 아닌 것을 다 공백으로 대체 ex) 숫자,특수문자 --> 공백 #sentence type?
        sentence = sentence.lower() #문자를 다 소문자로 변경

        tokens = []
        for ch in sentence:
            tokens.append(ord(ch) - ord('a') + 1) #단어에 있는 문자들을 숫자로 변경하여, 단어를 숫자로 변경
                                                 # embedding 층에 입력하기 위해서는 모두 정수로 인코딩 되어 있어야 함

        tokens = np.array(tokens) #각 단어를 np array로 바꿔줌
        return tokens

