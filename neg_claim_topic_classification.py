import logging
import os
import sys
from logging import getLogger

import pandas
import torch
from flair.embeddings import TransformerWordEmbeddings
from kbc.datasets import Dataset
from kbc.models import CP
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion

from classification import ClaimClassifier, EvaluationSetting
from embeddings import FlairTransformer, \
    ClamsKGGraphEmbeddingTransformer, NeighbourhoodVectorConcatStrategy, GraphEmbeddingTransformer

logger = getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == "__main__":
    args = sys.argv[1:]

    sparql_endpoint = args[0]

    num_splits = 10
    seed = 45345
    class_list = ["education", "healthcare", "immigration", "environment", "taxes", "elections", "crime"]
    claim_classifier = ClaimClassifier(class_list=class_list)

    dataset = pandas.read_csv(args[1], sep=",")

    # Concatenate claim and headline
    dataset['text'] = dataset[['text', 'headline']].apply(lambda x: ''.join(x), axis=1).to_list()

    # Concatenate all text
    dataset_text_all = dataset.copy()
    dataset_text_all['text'] = dataset[['text', 'headline', 'keywords', 'claim_date']].apply(lambda x: ''.join(x),
                                                                                             axis=1).to_list()
    # Only keywords
    dataset_kw = dataset.copy()
    dataset_kw['text'] = dataset[['keywords']].apply(lambda x: ''.join(x), axis=1).to_list()

    input_x = dataset[['claim', 'text']]
    input_x_all = dataset_text_all[['claim', 'text']]
    input_x_kw = dataset_kw[['claim', 'text']]

    input_y = dataset[class_list].copy().values

    data_path = args[2]
    model_path = args[3]

    # CKGE Graph embeddings
    ckge_dataset = Dataset(os.path.join(data_path, "CKGE"), use_cpu=True)
    ckge_model = CP(ckge_dataset.get_shape(), 50)
    ckge_model.load_state_dict(
        torch.load(os.path.join(model_path, "CKGE.pickle"),
                   map_location=torch.device('cpu')))

    ckge_graph_vectorizer = GraphEmbeddingTransformer(ckge_dataset, ckge_model)

    # Distil RoBERTa (DR)
    flair_vectorizer_DR = FlairTransformer([
        TransformerWordEmbeddings(model="distilroberta-base",
                                  use_scalar_mix=True)
    ], batch_size=1)

    # GPT2
    flair_vectorizer_GPT2 = FlairTransformer([
        TransformerWordEmbeddings(model="gpt2-large",
                                  use_scalar_mix=True)
    ], batch_size=1)

    ckge_dr_union = FeatureUnion([('TE', flair_vectorizer_DR), ('CP', ckge_graph_vectorizer)])
    ckge_gpt2_union = FeatureUnion([('TE', flair_vectorizer_GPT2), ('CP', ckge_graph_vectorizer)])

    # parametres_grid_ridge = {
    #     "estimator__alpha": [0.01, 0.1, 0.5, 1, 1.5, 3, 6],
    #     "estimator__normalize": [True, False],
    #     "estimator__tol": [1e-5, 1e-3, 1e-1]
    # }

    grid_search_params = {
        # "CKGE": parametres_grid_ridge,
    }

    print("Experiment 1: Complementarity of graph and text embedding feature...")
    experiment_1_settings = [
        EvaluationSetting("(1) CKGE",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_graph_vectorizer),
        EvaluationSetting("(2) TEDR",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=flair_vectorizer_DR),
        EvaluationSetting("(3) TEGPT2",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=flair_vectorizer_GPT2),
        EvaluationSetting("(1) CKGE & (2) TEDR",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_dr_union),
        EvaluationSetting("(1) CKGE & (3) TEGPT2",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gpt2_union),
    ]

    claim_classifier.evaluate(input_x, input_y, experiment_1_settings, n_folds=num_splits, seed=seed,
                              n_jobs=5, grid_search_params=grid_search_params)

    print("Experiment 2: Impact of != feature extraction strategies from graph embeddings...")

    ckge_graph_vectorizer_gnc = ClamsKGGraphEmbeddingTransformer(ckge_dataset, ckge_model, sparql_endpoint,
                                                                 NeighbourhoodVectorConcatStrategy.CONCAT_ALL,
                                                                 bidirectional=False)

    ckge_graph_vectorizer_gnt = ClamsKGGraphEmbeddingTransformer(ckge_dataset, ckge_model, sparql_endpoint,
                                                                 NeighbourhoodVectorConcatStrategy.CONCAT_TRIPLES,
                                                                 bidirectional=False)

    ckge_gnc_dr_union = FeatureUnion([('TE', flair_vectorizer_DR), ('CP', ckge_graph_vectorizer_gnc)])
    ckge_gnc_gpt2_union = FeatureUnion([('TE', flair_vectorizer_GPT2), ('CP', ckge_graph_vectorizer_gnc)])

    ckge_gnt_dr_union = FeatureUnion([('TE', flair_vectorizer_DR), ('CP', ckge_graph_vectorizer_gnt)])
    ckge_gnt_gpt2_union = FeatureUnion([('TE', flair_vectorizer_GPT2), ('CP', ckge_graph_vectorizer_gnt)])

    experiment_2_settings = [
        EvaluationSetting("(4) CKGE_GNC",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_graph_vectorizer_gnc),
        EvaluationSetting("(5) CKGE_GNT",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_graph_vectorizer_gnt),

        EvaluationSetting("(4) CKGE+GNC & (2) TEDR",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gnc_dr_union),
        EvaluationSetting("(4) CKGE+GNC & (3) TEGPT2",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gnc_gpt2_union),
        EvaluationSetting("(5) CKGE+GNT & (2) TEDR",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gnt_dr_union),
        EvaluationSetting("(5) CKGE+GNT & (3) TEGPT2",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gnt_gpt2_union),

    ]

    claim_classifier.evaluate(input_x, input_y, experiment_2_settings, n_folds=num_splits, seed=seed,
                              n_jobs=5, grid_search_params=grid_search_params)

    print("Ablation studies 1: Using only keywords for the text embedding...")

    ablation_1_settings = [
        EvaluationSetting("(6) TEDR-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=flair_vectorizer_DR),
        EvaluationSetting("(7) TEGPT2-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=flair_vectorizer_GPT2),
        EvaluationSetting("(1) CKGE & (6) TEDR-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_dr_union),
        EvaluationSetting("(1) CKGE & (7) TEGPT2-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gpt2_union),
        EvaluationSetting("(4) CKGE+GNC & (6) TEDR-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gnc_dr_union),
        EvaluationSetting("(4) CKGE+GNC & (7) TEGPT2-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gnc_gpt2_union),
    ]

    claim_classifier.evaluate(input_x_kw, input_y, ablation_1_settings, n_folds=num_splits, seed=seed,
                              n_jobs=5, grid_search_params=grid_search_params)

    print("Ablation studies 2: Graph embedding model without keywords...")

    # CKGE-KW Graph embeddings
    ckgekw_dataset = Dataset(os.path.join(data_path, "CKGE-KW"), use_cpu=True)
    ckgekw_model = CP(ckgekw_dataset.get_shape(), 50)
    ckgekw_model.load_state_dict(
        torch.load(os.path.join(model_path, "CKGE-KW.pickle"),
                   map_location=torch.device('cpu')))

    ckgekw_graph_vectorizer = GraphEmbeddingTransformer(ckgekw_dataset, ckgekw_model)
    ckgekw_graph_vectorizer_gnc = ClamsKGGraphEmbeddingTransformer(ckgekw_dataset, ckgekw_model, sparql_endpoint,
                                                                   NeighbourhoodVectorConcatStrategy.CONCAT_ALL,
                                                                   bidirectional=False)

    ckgekw_dr_union = FeatureUnion([('TE', flair_vectorizer_DR), ('CP', ckgekw_graph_vectorizer)])
    ckgekw_gpt2_union = FeatureUnion([('TE', flair_vectorizer_GPT2), ('CP', ckgekw_graph_vectorizer)])

    ckgekw_gnc_dr_union = FeatureUnion([('TE', flair_vectorizer_DR), ('CP', ckgekw_graph_vectorizer_gnc)])
    ckgekw_gnc_gpt2_union = FeatureUnion([('TE', flair_vectorizer_GPT2), ('CP', ckgekw_graph_vectorizer_gnc)])

    ablation_2_settings = [
        EvaluationSetting("(8) CPCKGE-KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_graph_vectorizer),
        EvaluationSetting("(9) CPCKGE+GNC-KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_graph_vectorizer_gnc),
        EvaluationSetting("(8) CKGE-KW & (6) TEDR-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_dr_union),
        EvaluationSetting("(8) CKGE-KW & (7) TEGPT2-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_gpt2_union),
        EvaluationSetting("(9) CKGE+GNC-KW & (6) TEDR-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_gnc_dr_union),
        EvaluationSetting("(9) CKGE+GNC-KW & (7) TEGPT2-C-H+KW",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_gnc_gpt2_union),
    ]
    claim_classifier.evaluate(input_x_kw, input_y, ablation_2_settings, n_folds=num_splits, seed=seed,
                              n_jobs=5, grid_search_params=grid_search_params)

    print("Ablation studies 3: Text embeddings of all text properties...")

    ablation_3_settings = [
        EvaluationSetting("(10) TEDR+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=flair_vectorizer_DR),
        EvaluationSetting("(11) TEGPT2+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=flair_vectorizer_GPT2),
        EvaluationSetting("(1) CKGE & (10) TEDR+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_dr_union),
        EvaluationSetting("(1) CKGE & (11) TEGPT2+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gpt2_union),
        EvaluationSetting("(4) CKGE+GNC & (10) TEDR+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gnc_dr_union),
        EvaluationSetting("(4) CKGE+GNC & (11) TEGPT2+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckge_gnc_gpt2_union),
        EvaluationSetting("(8) CKGE-KW & (10) TEDR+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_dr_union),
        EvaluationSetting("(8) CKGE-KW & (11) TEGPT2+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_gpt2_union),
        EvaluationSetting("(9) CKGE+GNC-KW & (10) TEDR+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_gnc_dr_union),
        EvaluationSetting("(9) CKGE+GNC-KW & (11) TEGPT2+KW+A",
                          MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
                          vectorizer=ckgekw_gnc_gpt2_union),
    ]

    claim_classifier.evaluate(input_x_all, input_y, ablation_3_settings, n_folds=num_splits, seed=seed,
                              n_jobs=5, grid_search_params=grid_search_params)
