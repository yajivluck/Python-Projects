# Markov Sentence Generator
==========================

This project uses a Markov model to generate random sentences based on a given vocabulary and sentence structure.

## Getting Started

### Prerequisites

* Python 3.x (tested on Python 3.8 and later)

### Installation

1. Clone the repository: `git clone https://github.com/yajivluck/Python-Projects.git`
2. Navigate to the project directory: `cd [path_to_cloned_project]`
3. Install the required dependencies: `pip install -r requirements.txt`

### Running the Project

1. Run the `main.py` file: `python main.py --vocab_file data/vocab.txt --unigram_counts_file data/unigram_counts.txt --bigram_counts_file data/bigram_counts.txt --trigram_counts_file data/trigram_counts.txt --num_sentences 10`

This will generate 10 random sentences using the Markov model. You can adjust the file paths and the number of sentences to generate as needed.

### Command-Line Options

* `--vocab_file`: Path to the vocabulary file (required)
* `--unigram_counts_file`: Path to the unigram counts file (required)
* `--bigram_counts_file`: Path to the bigram counts file (required)
* `--trigram_counts_file`: Path to the trigram counts file (required)
* `--num_sentences`: Number of sentences to generate (default: 5)

### Project Structure

* `main.py`: The entry point of the project, which uses the Markov model to generate sentences.
* `markov_model.py`: The Markov model implementation.
* `hmm.py`: The Hidden Markov Model implementation (not currently used).
* `data/`: Directory containing the vocabulary and sentence structure files.
* `requirements.txt`: List of dependencies required to run the project.

