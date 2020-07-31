import logging
import os
import sys
from logging import getLogger
import numpy
import pandas
import torch
from flair.embeddings import RoBERTaEmbeddings, DocumentPoolEmbeddings, TransformerWordEmbeddings
from kbc.datasets import Dataset
from kbc.models import CP
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from flair.embeddings import TransformerDocumentEmbeddings
from text_classification_2020 import ClaimClassifier, EvaluationSetting
from text_classification_2020.embeddings import GraphEmbeddingTransformer, FlairTransformer, GraphFlairTransformer
import csv

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
    class_list = ["education", "healthcare", "immigration", "environment", "taxes", "elections", "crime"]
    claim_classifier = ClaimClassifier(
        class_list=class_list)

    dataset = pandas.read_csv(args[0], sep=",")

    input_x_text = dataset['text'].to_list()
    input_x_graph = dataset['claim'].to_list()
    input_x = dataset[['claim', 'text']]
    #input_x['text'] = dataset[['text', 'headline']].apply(lambda x: ''.join(x), axis=1).to_list()

    input_y = dataset[class_list].copy().values

    # Graph embeddings
    dataset = Dataset(os.path.join(args[1]), use_cpu=True)
    model = CP(dataset.get_shape(), 50)
    model.load_state_dict(
        torch.load(args[2],
                   map_location=torch.device('cpu')))

    graph_vectorizer = GraphEmbeddingTransformer(dataset, model)

    #Baseline RoBERTa/BERT
    embeddings_baseline_roberta = [
        TransformerWordEmbeddings("roberta-base",layers="-1,-2,-3,-4",use_scalar_mix=True)
     ]
    document_embeddings_baseline_roberta = DocumentPoolEmbeddings(embeddings_baseline_roberta,
                                                                  fine_tune_mode="linear",
                                                                  pooling="mean")

    flair_vectorizer_baseline_roberta = FlairTransformer(document_embeddings_baseline_roberta)

    union_vectorizer = FeatureUnion([('flair', flair_vectorizer_baseline_roberta), ('graph', graph_vectorizer)])

    #Mean of the graph and BERT embeddings

    #mean_vectorizer=GraphFlairTransformer(document_embeddings_baseline_roberta,dataset, model)

    eval_settings = [

        # EvaluationSetting("cp_ckg_ridge",
        #                  MultiOutputClassifier(RidgeClassifier(normalize=True, fit_intercept=True, alpha=0.5)),
        #                  vectorizer= graph_vectorizer),
         EvaluationSetting("svm",
                           OneVsRestClassifier(estimator=SVC()),vectorizer=union_vectorizer),
        # EvaluationSetting("random_forest",RandomForestClassifier(),vectorizer=union_vectorizer),
    ]

    parametres_grid_ridge={
        "estimator__alpha": [0.01, 0.1, 0.5, 1, 1.5, 3, 6],
        "estimator__normalize": [True, False],
        "estimator__tol": [1e-5, 1e-3, 1e-1]
    }
    print("//////////////////::")
    print(RandomForestClassifier().get_params().keys())
    print("/////////////////:")

    param_grid={ 'estimator__C': numpy.logspace(-1, 2, 10),
        'estimator__kernel': ['poly','rbf','sigmoid'],
        'estimator__tol': [1e-5, 1e-3, 1e-1],
        'estimator__decision_function_shape': ['ovo','ovr'],
        'estimator__gamma': numpy.logspace(-1, 1, 10),}
    
    param_rf={
        'n_estimators': [100,200,300,400,500],
        'criterion': ["gini","entropy"],
        'max_features': ["auto","sqrt","log2"],
        'max_depth' : [4,5,6,7,8],
    }

    grid_search_params = {
        # "svm": param_grid
        #"roberta_baseline_ridge": parametres_grid_ridge
        "random_forest": param_rf
    }

    claim_classifier.evaluate(input_x, input_y, eval_settings, n_folds=num_splits, seed=seed,
                              n_jobs=5, grid_search_params=grid_search_params)

    # predictions=claim_classifier.get_predictions(input_x, input_y, eval_settings, n_folds=num_splits, seed=seed,n_jobs=5, grid_search_params=grid_search_params)
    # print("/////////////")
    # print(predictions.shape)
    # print(input_x.shape)
    # print(input_y.shape)
    # print("//////////////")

    # errors_index=claim_classifier.get_errors(predictions,input_y,which_label="education",labels=class_list)

    # for mislabeled_claim in errors_index:
    #     print("////////////////")
    #     print(input_x_text[mislabeled_claim])
    #     print(input_y[mislabeled_claim])
