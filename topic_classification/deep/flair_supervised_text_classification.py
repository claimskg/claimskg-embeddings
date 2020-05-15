#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:34:57 2019

@author: stagiaire
"""
import os

import pandas
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import DocumentRNNEmbeddings, CamembertEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from sklearn.metrics import confusion_matrix

print("Loading corpus...")

lemmatize = True
stop_word_filtering = True

# this is the folder in which train, test and dev files reside
data_folder = './deep/corpus'

# column format indicating which columns hold the text and label(s)
column_name_map = {1: "text", 2: "label_topic", }

# Camembert
camembert = CamembertEmbeddings(layers="-1,-2,-3,-4")

embedding_list = [camembert]

accuracies = list()

results = []
tps = []
tns = []
fps = []
fns = []

purposes = pandas.read_csv("motifs/purpose_for_training.csv")["name"].unique().tolist()

document_embeddings = DocumentRNNEmbeddings(embedding_list, hidden_size=150, reproject_words=True,
                                            reproject_words_dimension=300, bidirectional=True,
                                            rnn_layers=1,
                                            rnn_type='GRU',
                                            dropout=0.2,
                                            word_dropout=0.1)

for root, dirs, files in os.walk(data_folder):
    for dir in dirs:
        if "split" in dir:
            print("Processing " + dir + " ...")
            corpus: Corpus = ClassificationCorpus(data_folder + "/" + dir,
                                                  test_file='test.txt',
                                                  dev_file='dev.txt',
                                                  train_file='train.txt', in_memory=True)

            classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                        multi_label=False)
            trainer = ModelTrainer(classifier, corpus)
            # trainer = ModelTrainer(classifier, corpus)
            model_path = "./model/"
            scores = trainer.train(model_path, max_epochs=20,
                                   embeddings_storage_mode="cpu",
                                   learning_rate=0.5,
                                   mini_batch_size=2,
                                   anneal_factor=0.5,
                                   shuffle=True,
                                   patience=5, save_final_model=True, anneal_with_restarts=True)
            expected = [sentence.labels[0].value for sentence in corpus.test.sentences]
            predictions = [sentence.labels[0].value for sentence in classifier.predict(corpus.test.sentences)]
            cm = confusion_matrix(expected, predictions, purposes)
            tp = cm[1][1]
            tps.append(tp)
            tn = cm[0][0]
            tns.append(tn)
            fp = cm[0][1]
            fn = cm[1][0]

            print("TP={tp}; TN={tn}; FP={fp}; FN={fn}".format(tp=tp, tn=tn, fp=fp, fn=fn))
