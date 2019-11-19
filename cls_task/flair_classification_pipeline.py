import csv
import os
# from utils import get_class_labels
import sys
import traceback
from os import mkdir

from SPARQLWrapper import SPARQLWrapper
from flair.data import Corpus
# python cls_embed_claim_from_text.py ../../../data/entities.list ../../../data/claimskg.dataset.csv ../../../data/data_embeddings_utils/text_embeddings_claims
from flair.datasets import ClassificationCorpus
from flair.embeddings import DocumentRNNEmbeddings, RoBERTaEmbeddings
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
        claims = dict()
        for result in results:
            # check if ?o is a blank node (if yes, then iterate on it -- add on queue )
            id = result["claim"]["value"]
            rating_ = result["ratingName"]['value']
            text_ = str(result["text"]['value']).strip()
            if len(text_) > 0 and rating_ in labels:
                claims[id] = (text_.replace("\n", " ").replace("\t", "")[:511], rating_)
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

    mkdir(file_path + "/deep")

    for i in range(1, 10):
        fold_path = file_path + "/deep/split" + str(i)
        mkdir(fold_path)
        with open(file_path + "/split_test_" + str(i), "r") as test_fold:
            lines = test_fold.readlines()
            claims_list = []
            for line in lines:
                claims_list.append(claims[line.strip()])
            with open(fold_path + "/test.txt", mode="w") as f:
                write_fasttext_corpus(f, claims_list)

        with open(file_path + "/split_train_" + str(i), "r") as train_fold:
            lines = train_fold.readlines()
            claims_list = []
            for line in lines:
                claims_list.append(claims[line.strip()])

            train_end = int(0.9 * len(claims_list))

            with open(fold_path + "/train.txt", mode="w") as f:
                write_fasttext_corpus(f, claims_list[:train_end])

            with open(fold_path + "/dev.txt", mode="w") as f:
                write_fasttext_corpus(f, claims_list[train_end + 1:])

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
    compute = 'gpu'

    if args[0] == 'generate':
        generate_dataset(args[1], args[2].split(","))
    else:
        corpus_path = args[0]
        if len(args) > 1:
            compute = args[1]
        for root, dirs, files in os.walk(corpus_path):
            for dir in dirs:
                print("Processing " + dir + " ...")
                corpus: Corpus = ClassificationCorpus(corpus_path + "/" + dir,
                                                      test_file='test.txt',
                                                      dev_file='dev.txt',
                                                      train_file='train.txt', in_memory=True)

                # word_embeddings = [OpenAIGPT2Embeddings()]  # , RoBERTaEmbeddings(), XLNetEmbeddings(), OpenAIGPT2Embeddings()]
                word_embeddings = [RoBERTaEmbeddings()]

                document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True,
                                                            reproject_words_dimension=512, bidirectional=True,
                                                            rnn_layers=1,
                                                            rnn_type='GRU')

                classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                            multi_label=False)
                trainer = ModelTrainer(classifier, corpus)
                trainer.train('./' + dir + "/model/", max_epochs=20, embeddings_storage_mode=compute,
                              learning_rate=0.1,
                              mini_batch_size=32,
                              anneal_factor=0.5,
                              patience=5, save_final_model=True)
