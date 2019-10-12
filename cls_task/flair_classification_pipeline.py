import csv
# from utils import get_class_labels
import random
import sys
import traceback

import flair
from SPARQLWrapper import SPARQLWrapper
from flair.data import Corpus
# python cls_embed_claim_from_text.py ../../../data/entities.list ../../../data/claimskg.dataset.csv ../../../data/data_embeddings_utils/text_embeddings_claims
from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, RoBERTaEmbeddings, XLNetEmbeddings, OpenAIGPT2Embeddings, \
    DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from cls_task.sparql_offset_fetcher import SparQLOffsetFetcher

sparql_kg = SPARQLWrapper("http://localhost:8890/sparql")


def get_all_claims(labels):
    prefixes = """
PREFIX itsrdf: <https://www.w3.org/2005/11/its/rdf#>
PREFIX schema: <http://schema.org/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
    """
    where_body = """
    ?claim a schema:CreativeWork ; schema:text ?claim_text .
  ?claimReview schema:itemReviewed ?claim ; schema:reviewRating ?rating ; schema:headline ?headline.
  ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ; schema:alternateName ?ratingName 
    """

    fetcher = SparQLOffsetFetcher(sparql_kg, 10000, where_body=where_body,
                                  select_columns="?claim COALESCE(?claim_text, ?headline) as ?text ?ratingName",
                                  prefixes=prefixes)

    results = fetcher.fetch_all()

    # print(str_query)

    try:
        claims = []
        for result in results:
            # check if ?o is a blank node (if yes, then iterate on it -- add on queue )
            id = result["claim"]["value"]
            rating_ = result["ratingName"]['value']
            text_ = str(result["text"]['value']).strip()
            if len(text_) > 0 and rating_ in labels:
                claims.append((text_.replace("\n", " ").replace("\t", ""), rating_))
    except:
        print("Exception")
        print(traceback.format_exc())
        return (None, "NOT FOUND")
    return claims


def generate_dataset(file_path, labels):
    '''
    Generate a file storing all descriptions for all URIs
    '''
    print("generating data set: ", file_path)

    claims = get_all_claims(labels)
    random.shuffle(claims)

    train_end = int(0.7 * len(claims))
    test_start = train_end + 1
    test_end = int(test_start + 0.2 * len(claims))
    dev_start = test_end + 1

    with open(file_path + "/train.txt", mode='w') as f:
        write_fasttext_corpus(f, claims[:train_end])

    with open(file_path + "/test.txt", mode='w') as f:
        write_fasttext_corpus(f, claims[test_start:test_end])

    with open(file_path + "/dev.txt", mode='w') as f:
        write_fasttext_corpus(f, claims[dev_start:])

    print("data set generated")


def write_fasttext_corpus(file, claims):
    csv_writter = csv.writer(file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for claim in claims:
        csv_writter.writerow(["__label__" + claim[1], claim[0]])

    #
    # ############################################################################################################
    # # Transformer embeddings
    # # https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md
    # # + Elmo + Fastext (see en in simple conf)
    # ############################################################################################################
    # conf_trans = []
    #
    # conf_trans = [
    #     "BertEmbeddings",
    #     "OpenAIGPTEmbeddings",
    #     "OpenAIGPT2Embeddings",
    #     "TransformerXLEmbeddings",
    #     "XLNetEmbeddings",
    #     "XLMEmbeddings",
    #     "RoBERTaEmbeddings",
    #     "ELMoEmbeddings"
    # ]
    #
    # simple_conf = [
    #     "en-glove",
    #     "en-extvec",
    #     "en-crawl",
    #     "en-twitter",
    #     "en-turian",
    #     "en"  # FastText
    # ]
    #
    # configs = {
    #     "simple": simple_conf,
    #     "advanced": conf_trans
    # }
    #
    # return configs
    #


if __name__ == "__main__":

    args = sys.argv[1:]

    if args[0] == 'generate':
        generate_dataset(args[1], args[2].split(","))
    else:
        corpus_path = args[0]
        corpus: Corpus = ClassificationCorpus(corpus_path,
                                              test_file='test.txt',
                                              dev_file='dev.txt',
                                              train_file='train.txt')

        word_embeddings = [WordEmbeddings('en'), RoBERTaEmbeddings(), XLNetEmbeddings(), OpenAIGPT2Embeddings()]

        document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True,
                                                     reproject_words_dimension=256)

        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                    multi_label=False)
        trainer = ModelTrainer(classifier, corpus)
        trainer.train('./', max_epochs=10, embeddings_storage_mode='gpu')
