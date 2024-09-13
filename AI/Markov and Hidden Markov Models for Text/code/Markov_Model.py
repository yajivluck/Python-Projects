# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:19:40 2023

@author: yajiluck
"""

import random
import math
import numpy as np
from Levenshtein import distance as lev

#Lambda constant used in Poisson distribution for HMM Noise Correction
LAMBDA = 0.01


#Markov Model Class instantiation
class MarkovModel:

    #In this model, we use a vocab,unigram,bigram, and trigram txt file
    def __init__(self, vocab_file, unigram_file, bigram_file, trigram_file):
        self.vocab = self._load_vocab(vocab_file)
        self.unigram_probs = self._load_unigrams(unigram_file)
        self.bigram_probs = self._load_bigrams(bigram_file)
        self.trigram_probs = self._load_trigrams(trigram_file)

    #Helper function to parse vocab file
    
    #vocab is a dict where dict[index] = word.
    #Ex: vocab['501'] = 'Duty' (As a string)
    def _load_vocab(self, file):
        vocab_dict = {}
        with open(file, 'r') as f:
            for line in f:
                key, value = line.strip().split(' ', 1)
                vocab_dict[key] = value
        return vocab_dict


    #Helper function to parse unigram file
    
    
    #unigram_probs is a dictionary where each key corresponds to an index (i)
    #and each value corresponds to a float which represents P(Xt = i) in
    #for example self.unigram_probs[index = 500] = -5.358987
    
    #this means the probability that the next word in the sentence (on its own) being
    #vocab[499] = '500 During' = -5.358987. Note that vocab
    

    def _load_unigrams(self, file):
        unigrams = {}
        with open(file, 'r') as f:
            for line in f:
                word, prob = line.split()
                unigrams[word] = float(prob)
        return unigrams
    
    #Helper function to parse bigram file
    
    
    #self.bigram_counts is a dictionary of dictionaries.
    #The outer key of the dictionary is the index of a word i.
    #The inner key of the dictionary is the index of a word j
    
    
    #i j prob -> prob that next word is index j given current last word is index i
    
    #Probability that the next word is index j given the probability that current last
    #word is index i is found at self.bigram_counts['i']['j']
    
    def _load_bigrams(self, file):
        bigrams = {}
        with open(file, 'r') as f: 
            for line in f:
                w1, w2, prob = line.split()
                if w1 not in bigrams:
                    bigrams[w1] = {}
                bigrams[w1][w2] = float(prob)
        return bigrams

    #Helper function to parse trigram file
    
    #self.trigram_counts is a dictionary of dictionaries. The outer key
    #of the dictionary is a tuple of index strings such as ('100','10655')
    #which are respectively index i and j. The inner key of the dictionaries
    #are the indices k. The probability that the next word in a sentence is at index k
    #given that the current last two words are at index i and j is given by
    #self.trigram_counts[('i','j')]['k']
    def _load_trigrams(self, file):
        trigrams = {}
        with open(file, 'r') as f:
            for line in f:
                w1, w2, w3, prob = line.split()
                if (w1, w2) not in trigrams:
                    trigrams[(w1, w2)] = {}
                trigrams[(w1, w2)][w3] = float(prob)
        return trigrams

    #Markov Model Main function that generates sentences
    def generate_sentence(self):
        #Instantiate X0 = <s> (start of a sentence guaranteed.)
        #153 is index of <s> (start of sentence)
        sentence = ['153']
        
        while True:
            # Try trigram probabilities
            
            #Current last two words of the sentence being built
            prev_bigram = tuple(sentence[-2:])
            #If existing non-zero prob in trigram probability
            if prev_bigram in self.trigram_probs:
                #Get next word index based on trigram prob distributoin
                next_word = self._sample(self.trigram_probs[prev_bigram])
                #Given an existing next word
                if next_word:
                    #Add next word from trigram probs by index
                    sentence.append(next_word)
                    #If index is 152, break as '152' represents </s>
                    if next_word == '152':
                        break
                    continue
            # Fall back to bigram probabilities
            
            #Current last word of the sentence being built (by index)
            prev_word = sentence[-1]
            #If valid index on bigram
            if prev_word in self.bigram_probs:
                #Access bigram using previous word index as key
                next_word = self._sample(self.bigram_probs[prev_word])
                #If valid next word (existing non-zero prob from bigram)
                if next_word:
                    #Add next word by index from bigram probs
                    sentence.append(next_word)
                    #If end of sentence, break
                    if next_word == '152': #'152' is the key for </s> which is end of sentence.
                        break
                    continue
            # Fall back to unigram probabilities
            #Get index of unigram prob return
            next_word = self._sample(self.unigram_probs)
            if next_word:
                sentence.append(next_word)
                if next_word == '152':      #152 is index of </s> which is end of sentence
                    break
                
        
        #Translate list of indices into list of words
        translated_sentence = [self.vocab[index] for index in sentence]
        #Return list of words delimited by a space to form a sentence
        return ' '.join(translated_sentence)
    
    
    

    def _sample(self, probs_dict, randomness=True):
        if not randomness:
            # Directly return the word with the highest probability
            return max(probs_dict, key=probs_dict.get)
    
        total = sum(math.pow(10, p) for p in probs_dict.values())
        r = random.uniform(0, total)
        cumulative = 0
        for word, prob in probs_dict.items():
            cumulative += math.pow(10, prob)
            if r < cumulative:
                # Return word based on index
                return word
        return None
    
    



#Computes the log probability using the Poisson formula. Constant c is ommitted
def poisson_log_prob(k):
    return k * math.log10(LAMBDA) - math.log10(math.factorial(k))

#Computes the transition probability from word v to word u using
#Levenshtein distance k and poisson distribution
 
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

        # Base case initialization using vectorized operations
        transition_probs = np.array([self.bigram_probs['153'].get(s, float('-inf')) for s in self.vocab])
        emission_probs = np.array([transition_prob(observed_seq[0], self.vocab[s]) for s in self.vocab])
        
        viterbi[0] = transition_probs + emission_probs
        backpointer[0] = 0

        # Recursive step
        for t in range(1, T):
            for index,s in enumerate(self.vocab):
                trans_probs = viterbi[t-1] + np.array([self.bigram_probs.get(sp, {}).get(s, float('-inf')) for sp in self.vocab])
                
                max_index = np.argmax(trans_probs)
                max_trans_prob = trans_probs[max_index] + transition_prob(observed_seq[t], self.vocab[s])
                
                viterbi[t][index] = max_trans_prob
                backpointer[t][index] = max_index

        # Termination step
        best_last_state = np.argmax(viterbi[T-1])

        # Path backtracking
        best_path = [best_last_state]
        for t in range(T-1, 0, -1):
            best_last_state = backpointer[t][best_last_state]
            best_path.insert(0, best_last_state)

        return ' '.join([self.vocab[str(int(path) + 1)] for path in best_path])



if __name__ == "__main__":
      
    #Instantiate a markov object using the text files that are in the same directory
    markov = MarkovModel('vocab.txt', 'unigram_counts.txt', 'bigram_counts.txt', 'trigram_counts.txt')
    
    for i in range(5):
        print(markov.generate_sentence())
        
    # #Instantiate a hidden markov model object
    # hidden_markov = HiddenMarkovModel(vocab = markov.vocab, bigram_probs = markov.bigram_probs)
    
    # sentence_1 = "I think hat twelve thousand pounds"
    # sentence_2 = "she haf heard them"
    # sentence_3 = "She was ulreedy quit live"
    # sentence_4 = "John Knightly wasn't hard at work"
    # sentence_5 = "he said nit word by"
        
    # #Empty list that will hold sentences to correct noise
    # hidden_markov_inputs = []
    
    # sentences = [sentence_1,sentence_2,sentence_3,sentence_4,sentence_5]
    
    
    # hidden_markov_inputs.append(sentence_1.split())
    # hidden_markov_inputs.append(sentence_2.split())
    # hidden_markov_inputs.append(sentence_3.split())
    # hidden_markov_inputs.append(sentence_4.split())
    # hidden_markov_inputs.append(sentence_5.split())

    # for sentence_index,observed_seq in enumerate(hidden_markov_inputs):
        
    #     resolved_sentence = hidden_markov.viterbi(observed_seq)
        
    #     print('Noise input:', sentences[sentence_index],)
    #     print('\n')
    #     print('Resolved Input:', resolved_sentence)
    
    
    

    
    

