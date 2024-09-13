from utils import load_vocab, load_unigrams, load_bigrams, load_trigrams, sample

class MarkovModel:
    def __init__(self, vocab_file, unigram_file, bigram_file, trigram_file):
        self.vocab = load_vocab(vocab_file)
        self.unigram_probs = load_unigrams(unigram_file)
        self.bigram_probs = load_bigrams(bigram_file)
        self.trigram_probs = load_trigrams(trigram_file)

    def generate_sentence(self):
        sentence = ['153']  # Start of sentence
        while True:
            prev_bigram = tuple(sentence[-2:])
            if prev_bigram in self.trigram_probs:
                next_word = sample(self.trigram_probs[prev_bigram])
                if next_word:
                    sentence.append(next_word)
                    if next_word == '152':  # End of sentence
                        break
                    continue
            prev_word = sentence[-1]
            if prev_word in self.bigram_probs:
                next_word = sample(self.bigram_probs[prev_word])
                if next_word:
                    sentence.append(next_word)
                    if next_word == '152':  # End of sentence
                        break
                    continue
            next_word = sample(self.unigram_probs)
            if next_word:
                sentence.append(next_word)
                if next_word == '152':  # End of sentence
                    break
        
        translated_sentence = [self.vocab[index] for index in sentence]
        return ' '.join(translated_sentence)
