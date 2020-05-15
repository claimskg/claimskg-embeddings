from argparse import ArgumentParser

import gensim

parser = ArgumentParser(description="Converts word embeddings_baseline in text format to gensim")

parser.add_argument("--binary", dest="binary", action="store_true",
                    help="Set this flag if you are loading a binary embeddings_baseline file.", required=False)

parser.add_argument("input", help="Path to the input embeddings_baseline file", type=str, required=True)

parser.add_argument("output", help="Path of the output directory for the converted embeddings_baseline ", type=str,
                    required=True)

args = parser.parse_args()

word_vectors = gensim.models.KeyedVectors.load_word2vec_format(args.input, binary=args.binary)
word_vectors.save(args.output)
