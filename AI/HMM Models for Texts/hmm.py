import numpy as np
import math
from Levenshtein import distance as lev
from constants import LAMBDA

def poisson_log_prob(k):
    return k * math.log10(LAMBDA) - math.log10(math.factorial(k))

def transition_prob(u, v):
    k = lev(u, v)
    return poisson_log_prob(k)

class HiddenMarkovModel:
    def __init__(self, vocab, bigram_probs):
        self.vocab = vocab
        self.bigram_probs = bigram_probs

    def viterbi(self, observed_seq):
        T = len(observed_seq)
        N = len(self.vocab)
        
        viterbi = np.full((T, N), float('-inf'))
        backpointer = np.zeros((T, N), dtype=int)

        transition_probs = np.array([self.bigram_probs['153'].get(s, float('-inf')) for s in self.vocab])
        emission_probs = np.array([transition_prob(observed_seq[0], self.vocab[s]) for s in self.vocab])
        
        viterbi[0] = transition_probs + emission_probs
        backpointer[0] = 0

        for t in range(1, T):
            for index, s in enumerate(self.vocab):
                trans_probs = viterbi[t-1] + np.array([self.bigram_probs.get(sp, {}).get(s, float('-inf')) for sp in self.vocab])
                max_index = np.argmax(trans_probs)
                max_trans_prob = trans_probs[max_index] + transition_prob(observed_seq[t], self.vocab[s])
                
                viterbi[t][index] = max_trans_prob
                backpointer[t][index] = max_index

        best_last_state = np.argmax(viterbi[T-1])
        best_path = [best_last_state]

        for t in range(T-1, 0, -1):
            best_last_state = backpointer[t][best_last_state]
            best_path.insert(0, best_last_state)

        return ' '.join([self.vocab[str(int(path) + 1)] for path in best_path])
