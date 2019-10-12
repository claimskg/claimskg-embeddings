import getopt
import logging
import logging.config
import sys

import pandas as pd
# from utils import get_class_labels
from SPARQLWrapper import JSON, SPARQLWrapper
from redis import Redis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

sparql_kg = SPARQLWrapper("http://localhost:8890/sparql")
sparq_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")
logging.basicConfig(filename='app.log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')
# logging.config.fileConfig("log_config.ini", disable_existing_loggers=False)
logging.debug('Log @debug level')
logging.warning("Log @info level")
logging.warning('Log @warning level')
logging.error('Log @error level')
logging.critical('Log @critical level')

redis = Redis()


def get_class(claimID):
    result = None
    if redis:
        result = redis.get(claimID)
        if result is not None:
            return result
    if not redis or result is None:
        str_query = """
            PREFIX schema: <http://schema.org/>
            PREFIX nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
        
                SELECT ?text ?ratingName WHERE {
        
                <""" + claimID + """> schema:text ?text .
        
                ?claimReview schema:itemReviewed <""" + claimID + """> ; schema:reviewRating ?rating .
                ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ; schema:alternateName ?ratingName }
            """
        # print(str_query)
        # try:
        sparql_kg.setQuery(str_query)  # 1 false, 3 true
        sparql_kg.setReturnFormat(JSON)
        results = sparql_kg.query().convert()
        rating = str(-1)
        for result in results["results"]["bindings"]:
            # check if ?o is a blank node (if yes, then iterate on it -- add on queue )
            rating = str(result["ratingName"]['value'])

        if redis and rating is not None:
            redis.set(claimID, rating)
        # except:
        #     print("Exception -- Skipped claim ID" + str(claimID))
        # return "-1"
        return rating


def specify_models():
    nbayes = {'name': 'Naive Bayes', 'class': GaussianNB(), 'parameters': {}}

    knear = {
        'name': 'K Nearest Neighbors Classifier',
        'class': KNeighborsClassifier(),
        'parameters': {
            'n_neighbors': range(1, 12)
        }
    }

    svc_linear = {
        'name': 'Support Vector Classifier with Linear Kernel',
        'class': LinearSVC(),
        'parameters': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        }
    }

    sv_radial = {
        'name': 'Support Vector Classifier with Radial Kernel',
        'class': SVC(kernel='rbf'),
        'parameters': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        }
    }

    loglas = {
        'name': "Logistic Regression with LASSO",
        'class': LogisticRegression(penalty='l1'),
        'parameters': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        }
    }

    sgdc = {
        'name': "Stochastic Gradient Descent Classifier",
        'class': SGDClassifier(),
        'parameters': {
            'max_iter': [100, 1000],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        }
    }

    decis_tree = {
        'name': "Decision Tree Classifier",
        'class': DecisionTreeClassifier(),
        'parameters': {
            'max_depth': range(3, 15)
        }
    }

    ranfor = {
        'name': "Random Forest Classifier",
        'class': RandomForestClassifier(),
        'parameters': {
            'n_estimators': [10, 20, 50, 100, 200]
        }
    }

    extrerantree = {
        'name': "Extremely Randomized Trees Classifier",
        'class': ExtraTreesClassifier(),
        'parameters': {
            'n_estimators': [10, 20, 50, 100, 200]
        }
    }

    lis = list([
        nbayes, knear, svc_linear, sv_radial, loglas, sgdc, decis_tree, ranfor,
        extrerantree
    ])

    return lis


def subgraph2vec_tokenizer(s):
    '''
	Tokenize the string from subgraph2vec sentence (i.e. <target> <context1> <context2> ...). Just target is to be used
	and context strings to be ignored.
	:param s: context of graph2vec file.
	:return: List of targets from graph2vec file.
	'''
    return [line.split(' ')[0] for line in s.split('\n')]


def make_cls(model_dict, X, y, metric='f1', k=5):
    '''
		model_dict : We will pass in the dictionaries from the list you just created one by one to this parameter
		X: The input data
		y: The target variable
		metric : The name of a metric to use for evluating performance during cross validation. Please give this parameter a default value of 'f1' which is the F measure.
		k : The number of folds to use with cross validation, the default should be 5
		'''
    ''' # Choose the type of classifier.
	clf = RandomForestClassifier()

	# Choose some parameter combinations to try
	parameters = {
				  'max_depth': [2, 3],
				  }
	# Run the grid search
	grid_obj = GridSearchCV(clf, parameters, scoring='accuracy',cv= k)
	grid_obj = grid_obj.fit(X, y)

	best_parameters = grid_obj.best_params_
	print(best_parameters)
	best_score = grid_obj.best_score_
	print(best_score)
'''

    name = model_dict['name']
    param_grid = model_dict['parameters']
    logging.warning('MODEL ' + str(name))
    logging.warning('param_grid ' + str(param_grid))
    grid_obj = GridSearchCV(model_dict['class'],
                            param_grid,
                            scoring=metric,
                            cv=k,
                            refit='acc', n_jobs=8)
    grid_obj = grid_obj.fit(X, y)
    best_parameters = grid_obj.best_params_
    print(best_parameters)

    best_score = grid_obj.best_score_
    if False:
        logging.warning('BEST SCORE')
        logging.warning(best_score)
        logging.warning("CV results ")
        # aa = grid_obj.cv_results_
        logging.warning("MEAN TEST ACC")
        logging.warning(grid_obj.cv_results_['mean_test_acc'])
        logging.warning("MEAN TEST PREC")
        logging.warning(grid_obj.cv_results_['mean_test_prec_macro'])
        logging.warning("MEAN TEST REC")
        logging.warning(grid_obj.cv_results_['mean_test_rec_macro'])
        # best_score= clf.fit(X,y).best_score_
    best_model = grid_obj

    cls_app = model_dict['class']
    cls_app.set_params(**best_parameters)
    scoring = ['precision_macro', 'recall_macro', 'accuracy']
    results_kfold = cross_validate(cls_app,
                                   X,
                                   y,
                                   scoring=scoring,
                                   cv=k,
                                   return_train_score=False)
    logging.warning("RESULT K FOLD")
    logging.warning(results_kfold)
    logging.warning("MEAN TEST ACC K FOLD")
    logging.warning(results_kfold['test_accuracy'])
    logging.warning("MEAN TEST PREC K FOLD")
    logging.warning(results_kfold['test_recall_macro'])
    logging.warning("MEAN TEST REC K FOLD")
    logging.warning(results_kfold['test_precision_macro'])
    logging.warning("\n")
    logging.warning("\n")

    return (name, best_model, best_score)


def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (accuracy_score(y, yPred),
            precision_score(y, yPred, pos_label=3, average='macro'),
            recall_score(y, yPred, pos_label=3, average='macro'))


def my_scorer_funct(estimator, x, y):
    a, p, r = getScores(estimator, x, y)
    return a, p, r


if __name__ == '__main__':
    precreated_file_ready = True
    input_path = None
    output_dataframe_path = None
    input_dataframe_path = None
    exclusion_file_path = None

    true_vs_false = True
    true_and_false_vs_mix = False

    text_input_features = False
    exclusion_list = []

    # try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["generate-dataframe=", "input-features=", "dataframe=",
                                                  "true-false-mixed",
                                                  "text-input-features", "error-file"])
    for opt, arg in opts:
        if opt == '--generate-dataframe':
            precreated_file_ready = False
            output_dataframe_path = str(arg)
        if opt == '--input-features':
            if precreated_file_ready:
                logging.error(
                    "--input-features required for --generate-dataframe")
                exit(1)
            input_path = str(arg)
        if opt == '--exclusion-file':
            exclusion_file_path = str(arg)
            exclusion_file = open(exclusion_file_path, "r", encoding="utf8")
            exclusion_list = exclusion_file.readlines()
        if opt == '--text-input-features':
            text_input_features = True
        if opt == '--dataframe':
            input_dataframe_path = str(arg)
        if opt == '--true-false-mixed':
            true_vs_false = False
            true_and_false_vs_mix = True

    # except:
    #     print('Arguments parser error, try -h')
    #     exit()

    # Reading the Data and Performing Basic Data Checks
    # tsv_read = pd.read_csv(file_path + file_name, sep='\t')
    if not precreated_file_ready:
        if text_input_features:
            sep = ","
        else:
            sep = "\t"
        f = open(input_path, 'r', encoding='utf8')
        lines = f.readlines()
        parts = lines[0].split(sep)

        if text_input_features:
            dims = len(parts) - 1
        else:
            dims = len(parts) - 1

        # create a list of col names
        cnames = ['nodeID']
        for i in range(0, dims-1):
            cnames.append("feature" + str(i))
        cnames.append('target')

        # Creating vectors for features and class/response Variable
        df = pd.DataFrame(columns=cnames)
        list_of_lists = []
        i_count = 0

        for line in tqdm(lines):
            # print(line.translate(table), end="")
            parts = line.strip().split(sep)
            if parts[0].startswith("<"):
                parts[0] = parts[0][1:-1]
            if not (parts[0].startswith("http://data.gesis.org/claimskg/creative_work/")):
                continue
            parts[1:dims] = [float(part) for part in parts[1:dims]]

            if not text_input_features:
                line_class = get_class(parts[0])
                parts.append(line_class)
            else:
                line_class = parts[-1]

            if true_and_false_vs_mix and (
                    line_class == 'MIXTURE' or line_class == 'TRUE' or line_class == 'FALSE' and parts[
                0] not in exclusion_list):
                list_of_lists.append(parts)
            elif line_class == 'TRUE' or line_class == 'FALSE' and parts[0] not in exclusion_list:
                list_of_lists.append(parts)
        df = df.append(pd.DataFrame(list_of_lists, columns=df.columns))
        df.to_csv(output_dataframe_path, sep=',')
    else:
        # read file already preprared with feat and target
        df = pd.read_csv(input_dataframe_path, sep='\t', encoding='utf-8')

    # drop claim where claim ID not recognized
    df = df[df.target != -1]
    if true_vs_false:
        df = df[df.target != 2]
    if true_and_false_vs_mix:
        df.loc[df['target'] == 3, 'target'] = 1
    data = df
    x_feature_vectors = data.drop('target', axis=1).drop('nodeID',
                                                         axis=1).values
    print(x_feature_vectors.shape)
    y_class_vector = data['target'].values
    print(y_class_vector.shape)
    print(df['target'].value_counts())
    # test Different Model using KFOLD
    models_dict = dict()
    model_list = specify_models()

    my_scores = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro'
    }

    kfold = KFold(n_splits=10, random_state=100)
    for model_dict in model_list:
        print("Cross Validation for " + str(model_dict['name']))
        print(
            make_cls(model_dict,
                     x_feature_vectors,
                     y_class_vector,
                     my_scores,
                     k=10))

    pass
