import argparse
from markov_model import MarkovModel
# from hmm import HiddenMarkovModel

def main():
    parser = argparse.ArgumentParser(description='Generate sentences using a Markov model')
    parser.add_argument('--vocab_file', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--unigram_counts_file', type=str, required=True, help='Path to unigram counts file')
    parser.add_argument('--bigram_counts_file', type=str, required=True, help='Path to bigram counts file')
    parser.add_argument('--trigram_counts_file', type=str, required=True, help='Path to trigram counts file')
    parser.add_argument('--num_sentences', type=int, default=5, help='Number of sentences to generate')
    args = parser.parse_args()

    markov = MarkovModel(args.vocab_file, args.unigram_counts_file, args.bigram_counts_file, args.trigram_counts_file)

    for i in range(args.num_sentences):
        print(markov.generate_sentence())

    # hidden_markov = HiddenMarkovModel(vocab=markov.vocab, bigram_probs=markov.bigram_probs)
    # sentences = ["I think hat twelve thousand pounds", ...]
    # for sentence in sentences:
    #     print(hidden_markov.viterbi(sentence.split()))

if __name__ == "__main__":
    main()