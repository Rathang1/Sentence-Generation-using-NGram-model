import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    START = '<s>'
    END = '</s>'

    new_text = [START]*(n-1)
    new_text.extend(text)
    new_text.append(END)

    for i in range(n-1,len(new_text)):
        trail = new_text[i-n+1:i]
        yield tuple((new_text[i],tuple(trail)))


# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    file = open(corpus_path).read()
    paras = file.split('\n\n')

    sentences = []
    for para in paras:
        sentences.extend(sent_tokenize(para))

    words = []
    for sentence in sentences:
        words.append(word_tokenize(sentence))

    # print(words)

    return words



# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    data = load_corpus(corpus_path)
    ngram_lm = NGramLM(n)

    for i in data:
        ngram_lm.update(i)
    return ngram_lm


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        self.vocabulary |= set(text)
        self.vocabulary |= set(['</s>'])
        for ngram in get_ngrams(self.n, text):
            self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
            self.context_counts[ngram[1]] = self.context_counts.get(ngram[1], 0) + 1


    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        input1 = (word, context)
        context_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts.get(input1, 0)
        result = 0
        if delta == 0:
            if context_count == 0:
                return (1/(len(self.vocabulary)))
            else:
                return ngram_count/context_count

        else:
            return (ngram_count + delta)/(context_count + delta*len(self.vocabulary))

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        sum1 = 0
        for ngram in get_ngrams(self.n, sent):
            res = self.get_ngram_prob(ngram[0], ngram[1], delta)
            try:
                sum1 += math.log2(res)
            except:
                sum1 += float('-inf')

        return sum1

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:
        corpusLen = 0
        corpusProb = 0

        for i in corpus:
            corpusLen = corpusLen + len(i)
            currSentProb = self.get_sent_log_prob(i)
            corpusProb = corpusProb + currSentProb

        avg_log_prob = (-1) * (corpusProb / corpusLen)
        return math.pow(2, avg_log_prob)

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        newVocab = sorted(self.vocabulary)
        r = random.random()
        prevProbability = 0
        currProbability = 0
        for word in newVocab:
            ngramProbability = self.get_ngram_prob(word, context, delta)
            currProbability = currProbability + ngramProbability
            if prevProbability < r <= currProbability:
                return word
            prevProbability = prevProbability + ngramProbability

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        currentLength = 0
        res = ['<s>'] * (self.n - 1)
        first_word = self.generate_random_word(tuple(res), delta)
        currentLength = currentLength + 1
        res.append(first_word)
        sentEnd = False
        while currentLength < max_length and not sentEnd:
            word = self.generate_random_word(tuple(res[-(self.n - 1):]), delta)
            res.append(word)
            currentLength = currentLength + 1
            if word == '</s>':
                sentEnd = True

        return ' '.join(res[self.n - 1:])


def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(3, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'

    print(trigram_lm.get_sent_log_prob(word_tokenize(s1)))
    print(trigram_lm.get_sent_log_prob(word_tokenize(s2)))
    print('test')
    print(trigram_lm.generate_random_text(10,0.4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
