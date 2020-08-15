import json
import os
import pickle
from copy import deepcopy
from logging import getLogger
from typing import List

import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.pipeline import make_pipeline, Pipeline
from tqdm import tqdm

logger = getLogger()

class EvaluationSetting:
    """
    This class allows to define a particular evaluation setting that combines a vectorizer (VectorMixin) and a
    classifier.
    """

    def __init__(self, name, classifier, vectorizer=None):
        """
        When using a scikit-lean component (TfidfVectorizer, CountVectorizer, Binarizer, etc.), it is recommended to
        **not** set a vectorizer here, but to use a scikit-learn pipeline in order to enable a joint parameter
        estimation of the vectorizer and of the classifier and to make sure the vectorizer is fitted only with the
        appropriate data during grid search and crossvalidation.
        The one caveat is that one cannot use TfidfVectorizer directly before the classifier: one must
        start with a CountVectorizer followed by a TfidfTransformer and then the classifier/estimator.
        This example,
        >>> pipeline = make_pipeline(("count_vectorizer", CountVectorizer()), ("tfidf_tran", TfidfTransformer()),
        >>>                           ("classifier", LogisticRegression()))
        is strictly equivalent to a TfidfVectorizer() followed by LogisticRegression(), however pieplines require
        and intermediary TransformerMixin.

        However, custom vectorizers (e.g. FlairVectorizer), cannot be used in a pipeline

        Parameters
        ----------
        name The name of the evaluation setting (used for display and identification)
        vectorizer A scikit-learn transformer for vectorization (TransformerMixin): i.e. transforming the textual dataset
        into vectors. Optional, Default None
        classifier A scikit-learn BaseEstimator used for classification
        """
        self.name = name
        self.vectorizer = vectorizer
        self.classifier = classifier

        self.scores = []
        self.mean_scores = {}
        self.std_scores = {}

    def __str__(self) -> str:
        return "{},{},{}".format(self.name, self.mean_scores, self.std_scores)


class ClaimClassifier:
    """
        This class represents a classification system for the ESII problem
    """

    def __init__(self, class_list,
                 scoring=None):
        """
        Parameters
        ----------
        class_list: List[str]
        List of classes for the classification task
        """
        # Build an index of classes so that each label is associated to an unique integer
        # self.scoring = [('accuracy', 'accuracy'),
        #                 ('f1', 'f1_samples'), ('P', 'precision_samples'),
        #                 ('R', 'recall_samples'), ('Pm', 'precision_micro'),
        #                 ('Rm', 'recall_micro'), ('PM', 'precision_macro'),
        #                 ('RM', 'recall_macro'), ('f1m', 'f1_micro'), ('f1M', 'f1_macro')]
        if scoring is None:
            scoring = ['accuracy', 'f1_samples', 'precision_samples', 'recall_samples',
                       'precision_micro',
                       'recall_micro', 'precision_macro',
                       'recall_macro', 'f1_micro', 'f1_macro']
        self.scoring = scoring
        self._index_classes(class_list)
        self.vectorizer = None
        self.classifier = None

    def _index_classes(self, classes):
        self.class_index = {}
        self.reverse_class_index = {}
        current_index = 0
        for cls in classes:
            self.class_index[cls] = current_index
            self.reverse_class_index[current_index] = cls
            current_index += 1

    def _build_vector_model(self, input, vectorizer=CountVectorizer()):
        self.vectorizer = vectorizer
        logger.info("Fitting vectorizer [{}]".format(str(self.vectorizer.__class__.__name__)))
        self.vectorizer.fit(input)

    def _vectorize(self, input_x):
        X = self.vectorizer.transform(input_x)
        return X

    def train(self, classifier, input_x, input_y, vectorizer=None, model_directory="model"):
        """
                Trains the model using a scikit-learn Pipeline that specifies a series of feature transformers
                followed by a classifier.

                Note: If you run evaluate, the best EvaluationSetting will be set as the pipeline
                of this instance of ClaimClassifier, and you won't need to run train again before using classify.


                Parameters
                ----------
                classifier: BaseEstimator
                    The scikit-learn estimator/classifier
               input_x_text: List[str]
                    The input, where each element is a request in text form
                input_y: List[str]
                    The expected classification results, as a list of purposes (class names)
                model_file: str
                    (Optional) Path to a serialized model

                Returns
                -------
        """
        # Replace dataset labels by their corresponding indices
        y = numpy.array([self.class_index[label] for label in input_y])

        use_pipeline = isinstance(classifier, Pipeline)
        self.classifier = classifier

        if not use_pipeline:
            X = self._vectorize(input_x)
            logger.info("Fitting classifier [{}]...".format(str(self.classifier.__class__.__name__)))
            self.classifier.fit(X, y)
        else:
            logger.info("Fitting classifier [{}]...".format(str(self.classifier.__class__.__name__)))
            self.classifier.fit(input_x, y)

        self._save_model(classifier, model_directory)

    @staticmethod
    def _save_model(estimator, model_file="model.pkl"):
        logger.info("Saving model [{}]...".format(model_file))
        with open(model_file, 'wb') as fid:
            pickle.dump(estimator, fid)

    def load_classifier(self, model_directory="model.pkl"):
        """
                Loads a serialized classification pipeline

                Parameters
                ----------
                model_directory: str
                    (Optional) The path to the directory that contains the serialized model, the vectorizers and
                    the overall configuration

                Returns
                -------

        """
        if not os.path.exists(model_directory) or os.path.isdir(model_directory):
            logger.error("FATAL: Model directory does not exist!")
            exit(1)

        if not os.path.exists(model_directory + "/config.json"):
            logger.error("FATAL: Model configuration not found: " + str(model_directory + "/config.json"))
            exit(1)

        config = json.load(open(model_directory + "/config.json", "r"))

        logger.info("Loading classifier [from {}]...".format(model_directory + "/model.pkl"))
        with open(model_directory + "/model.pkl", "rb") as fid:
            self.classifier = pickle.load(fid)

    def evaluate(self, input_x, input_y, evaluation_settings: List[EvaluationSetting], n_folds=10,
                 seed=100, grid_search_params=None, save_model=False, model_directory="model.pkl", n_jobs=-1):
        """
                This method takes the dataset as input and a list of EvaluationSettings and performs k-fold cross validation
                on each evaluation setting, after which the pipeline for the best setting is set as the classifier for this
                instance of ClaimClassifier.

                Parameters
                ----------
               input_x_text: List[str]
                    The input, where each element is a request in text form
               input_y: List[str]
                    The expected classification results, as a list of purposes (class names)
               evaluation_settings: List[EvaluationSetting]
                   A list of EvaluationSetting instances that wrap the scikit learn Pipelines
               n_folds: int
                   The number of folds to use. Default: 10
               seed: int
                   The seed for the random state of the KFold splitting. Default: 100
               grid_search_dict: List[Dict]
                    A dictionary of parameter dictionaries for each evaluation setting for GridSearchCV.
                    The key for the parameters for a given evaluation setting should be the
                    same as the name of the evaluation setting name.
                    If provided a grid search will be executed before the final cross validation on the basis of the
                    provided parameters.
                    Optimal parameters found by the grid search will be used for the cross validation.
                save_model: bool
                    If True, serialize the best model across grid search optimal parameters and evaluation settings.
                    By default model_file specifies the name of the file where the model should be saved.
                model_file: str
                    Path where the model should be saved, by default ./model.pkl. Only used is save_model is set to
                    True
                n_jobs: int
                    number of joblib jobs to use during cross validation or grid search cv. Default -1 (all cores).


               Returns
               -------

               List[EvaluationSetting]
                   The list of evaluation settings, where the score, mean_scores and std_scores attributes contain the
                   corresponding evaluation scores
        """

        # Fixing the seed, ensures the folds will be the same every time (ensures comparable results)
        stratif = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        # Replace dataset labels by their corresponding indices
        y = input_y

        # Keep a dictionary of unique vectorizers where the vectorized dataset is stored for each different instance of
        # the vectorizer. This way if several settings share the same vectorizer, the computation won't be duplicated
        vectorizer_instance_dict = {}

        best_setting = None
        best_estimator_index = 0
        best_score = 0
        current_estimator_index = 0

        for setting in tqdm(evaluation_settings):

            input_x_remaining, input_x_gscv, y_remaining, y_gscv = train_test_split(input_x, y, random_state=seed,
                                                                                    shuffle=True,
                                                                                    train_size=0.8)

            use_pipeline = isinstance(setting.classifier, Pipeline)

            classifier = setting.classifier

            if grid_search_params is not None and setting.name in grid_search_params.keys():
                # Grid Search parameters supplied, we perform grid-search on 20% held-out data
                classifier_clone = deepcopy(classifier)
                grid_cv = GridSearchCV(classifier_clone, grid_search_params[setting.name], n_jobs=n_jobs, cv=3)

                if use_pipeline:
                    grid_cv = grid_cv.fit(input_x_gscv, y_gscv)
                else:
                    X_gscv = deepcopy(setting.vectorizer).fit_transform(input_x_gscv)
                    grid_cv = grid_cv.fit(X_gscv, y_gscv)

                classifier.set_params(**grid_cv.best_params_)
                logger.info("Best parameters: " + str(grid_cv.best_params_))

            classifier_clone = deepcopy(classifier)
            if use_pipeline:
                scores = cross_validate(classifier_clone, input_x_remaining, y_remaining, cv=stratif,
                                        return_estimator=True, scoring=self.scoring)
            else:
                if setting.vectorizer in vectorizer_instance_dict.keys():
                    X_remaining = vectorizer_instance_dict[setting.vectorizer]
                else:
                    X_remaining = deepcopy(setting.vectorizer).fit_transform(input_x_remaining)
                    if not isinstance(X_remaining, numpy.ndarray):
                        X_remaining = X_remaining.toarray()
                    vectorizer_instance_dict[setting.vectorizer] = X_remaining

                scores = cross_validate(classifier_clone, X_remaining, y_remaining, cv=stratif, return_estimator=True,
                                        scoring=self.scoring)

            setting.scores = {}
            setting.mean_scores = {}
            setting.std_scores = {}
            for scoring in self.scoring:
                key = "test_" + scoring
                scores_vector = scores[key]
                setting.scores[key] = scores_vector
                setting.mean_scores[key] = scores_vector.mean()
                setting.std_scores[key] = scores_vector.std()

            logger.info(setting)

            accuracy_key = "test_accuracy"
            mean_acc = setting.mean_scores[accuracy_key]
            std_acc = setting.std_scores[accuracy_key]
            if mean_acc - std_acc > best_score:
                best_score = mean_acc - std_acc
                best_setting = setting
                best_estimator_index = current_estimator_index
                current_estimator_index += 1

        if best_setting is not None:
            logger.info("Best estimator overall: " + str(best_setting.classifier) + " with score:" + str(
                evaluation_settings[best_estimator_index].mean_scores))
            if save_model:
                self._build_vector_model(input_x, vectorizer=best_setting.vectorizer)
                self.train(best_setting.classifier, input_x, y, model_directory=model_directory, )

    def classify(self, input_x: List[str]):
        """
        Perform classification on an input dataset (list of strings)
        Parameters
        ----------
        input_x_text a List[str] containing the requests to classify

        Returns
        -------
        predictions: List[str]
        The predictions as a list of strings where each element is the purpose for the corresponding request

        """
        if self.vectorizer is not None and self.classifier is not None:
            X = self._vectorize(input_x)
            logger.info("Classifying...")
            predictions = self.classifier.predict(X)
            return [self.reverse_class_index[prediction] for prediction in predictions]

        return None
