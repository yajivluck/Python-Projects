import math
import random

def load_vocab(file):
    vocab_dict = {}
    with open(file, 'r') as f:
        for line in f:
            key, value = line.strip().split(' ', 1)
            vocab_dict[key] = value
    return vocab_dict

def load_unigrams(file):
    unigrams = {}
    with open(file, 'r') as f:
        for line in f:
            word, prob = line.split()
            unigrams[word] = float(prob)
    return unigrams

def load_bigrams(file):
    bigrams = {}
    with open(file, 'r') as f: 
        for line in f:
            w1, w2, prob = line.split()
            if w1 not in bigrams:
                bigrams[w1] = {}
            bigrams[w1][w2] = float(prob)
    return bigrams

def load_trigrams(file):
    trigrams = {}
    with open(file, 'r') as f:
        for line in f:
            w1, w2, w3, prob = line.split()
            if (w1, w2) not in trigrams:
                trigrams[(w1, w2)] = {}
            trigrams[(w1, w2)][w3] = float(prob)
    return trigrams

def sample(probs_dict, randomness=True):
    if not randomness:
        return max(probs_dict, key=probs_dict.get)

    total = sum(math.pow(10, p) for p in probs_dict.values())
    r = random.uniform(0, total)
    cumulative = 0
    for word, prob in probs_dict.items():
        cumulative += math.pow(10, prob)
        if r < cumulative:
            return word
    return None
