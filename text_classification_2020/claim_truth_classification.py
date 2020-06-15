import logging
import os
import sys
from logging import getLogger

import torch
from SPARQLWrapper import SPARQLWrapper
from flair.embeddings import RoBERTaEmbeddings, DocumentPoolEmbeddings
from kbc.datasets import Dataset
from kbc.models import CP
from pandas import DataFrame
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import FeatureUnion

from ckge.utils import get_all_claims
from text_classification_2020 import ClaimClassifier, EvaluationSetting
from text_classification_2020.embeddings import GraphEmbeddingTransformer, FlairTransformer

logger = getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == "__main__":
    args = sys.argv[1:]

    num_splits = 10
    seed = 45345
    classes = ["TRUE", "FALSE", "MIXTURE"]
    claim_classifier = ClaimClassifier(class_list=classes, scoring=['accuracy', 'precision_micro',
                                                                    'recall_micro', 'precision_macro',
                                                                    'recall_macro', 'f1_micro', 'f1_macro'])

    sparql_kg = SPARQLWrapper(args[0])

    all_claims = get_all_claims(sparql_kg, classes)
    claims = []
    texts = []
    ratings = []

    for claim in all_claims:
        values = all_claims[claim]
        claims.append(claim)
        texts.append(values[0])
        ratings.append(values[1])

    input_x = DataFrame()
    input_x['claim'] = claims
    input_x['text'] = texts

    # input_x_text = dataset[['text', 'headline']].apply(lambda x: ''.join(x), axis=1).to_list()

    input_y = ratings

    # Graph embeddings
    dataset = Dataset(os.path.join(args[1]), use_cpu=True)
    model = CP(dataset.get_shape(), 50)
    model.load_state_dict(
        torch.load(args[2],
                   map_location=torch.device('cpu')))

    graph_vectorizer = GraphEmbeddingTransformer(dataset, model)

    # Baseline RoBERTa/BERT
    embeddings_baseline_roberta = [
        RoBERTaEmbeddings(pretrained_model_name_or_path="distilroberta-base",
                          use_scalar_mix=False)
    ]
    document_embeddings_baseline_roberta = DocumentPoolEmbeddings(embeddings_baseline_roberta,
                                                                  fine_tune_mode="linear",
                                                                  pooling="mean")
    flair_vectorizer_baseline_roberta = FlairTransformer(document_embeddings_baseline_roberta)

    union_vectorizer = FeatureUnion([('flair', flair_vectorizer_baseline_roberta), ('graph', graph_vectorizer)])

    eval_settings = [
        # EvaluationSetting("roberta_baseline_ridge",
        #                   MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
        #                   vectorizer=flair_vectorizer_baseline_roberta),
        EvaluationSetting("TrC-CP_CKGE",
                          RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5),
                          vectorizer=graph_vectorizer),
        # EvaluationSetting("TrC-DistilRoberta",
        #                   RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5),
        #                   vectorizer=flair_vectorizer_baseline_roberta),
        EvaluationSetting("TrC-DistilRoberta-CP_CKG",
                          RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5),
                          vectorizer=union_vectorizer),
    ]

    parametres_grid_ridge = {
        "estimator__alpha": [0.01, 0.1, 0.5, 1, 1.5, 3, 6],
        "estimator__normalize": [True, False],
        "estimator__tol": [1e-5, 1e-3, 1e-1]
    }

    grid_search_params = {
        "roberta_baseline_ridge": parametres_grid_ridge
    }

    claim_classifier.evaluate(input_x, input_y, eval_settings, n_folds=num_splits, seed=seed,
                              n_jobs=5, grid_search_params=grid_search_params)
