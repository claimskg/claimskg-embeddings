import getopt
import logging
import logging.config
import sys

from sklearn.utils import resample
from cls_task import scoring_functions
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
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

sparql_kg = SPARQLWrapper("http://localhost:8890/sparql")
sparq_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")
logging.basicConfig(filename='app.log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')
# logging.config.fileConfig("log_config.ini", disable_existing_loggers=False)
##logging.debug('Log @debug level')
##logging.info("Log @info level")
#logging.warning('Log @warning level')
#logging.error('Log @error level')
#logging.critical('Log @critical level')
logging.warning("New run")
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

    nn_mlp = {
        'name': 'multi-layer perceptron',
        'class': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        'parameters': {
            'hidden_layer_sizes': [(5, 2), (3, 2)]
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
                            refit='accuracy', n_jobs=8)
    grid_obj = grid_obj.fit(X, y)
    best_parameters = grid_obj.best_params_
    print(best_parameters)

    best_score = grid_obj.best_score_
        # best_score= clf.fit(X,y).best_score_
    best_model = grid_obj

    cls_app = model_dict['class']
    cls_app.set_params(**best_parameters)
    #scoring = ['precision_macro', 'recall_macro', 'accuracy']

    scoring = scoring_functions.overall_scoring()
    results_kfold = cross_validate(cls_app, X, y, scoring=scoring, cv=k, return_train_score=False)

    str_out = ""
    acc_list = list(results_kfold['test_accuracy'])
    str_out +=  str(sum(acc_list) / len(acc_list)) + "\t"

    f1_list = list(results_kfold['test_f1'])
    str_out += str(sum(f1_list) / len(f1_list)) + "\t"
    prec_list = list(results_kfold['test_precision'])
    str_out += str(sum(prec_list) / len(prec_list)) + "\t"
    rec_list = list(results_kfold['test_recall'])
    str_out += str(sum(rec_list) / len(rec_list)) + "\t"

    f1_list = list(results_kfold['test_f1_macro'])
    str_out += str(sum(f1_list) / len(f1_list)) + "\t"
    prec_list = list(results_kfold['test_precision_macro'])
    str_out += str(sum(prec_list) / len(prec_list)) + "\t"
    rec_list = list(results_kfold['test_recall_macro'])
    str_out += str(sum(rec_list) / len(rec_list)) + "\t"

    f1_list = list(results_kfold['test_f1_micro'])
    str_out += str(sum(f1_list) / len(f1_list)) + "\t"
    prec_list = list(results_kfold['test_precision_micro'])
    str_out += str(sum(prec_list) / len(prec_list)) + "\t"
    rec_list = list(results_kfold['test_recall_micro'])
    str_out += str(sum(rec_list) / len(rec_list)) + "\t"

    tp_list = list(results_kfold['test_tp'])
    str_out += str(sum(tp_list) / len(tp_list)) + "\t"
    tn_list = list(results_kfold['test_tn'])
    str_out += str(sum(tn_list) / len(tn_list)) + "\t"

    fp_list = list(results_kfold['test_fp'])
    str_out += str(sum(fp_list) / len(fp_list)) + "\t"
    fn_list = list(results_kfold['test_fn'])
    str_out += str(sum(fn_list) / len(fn_list)) + "\t"

    logging.warning("+++++++++++++++++++++")
    logging.warning(str_out)
    logging.warning("+++++++++++++++++++++")
    print("number of folds " + str(k))
    '''for i in range(0, k):
        f1 = 2 * (rec_list[i] * prec_list[i]) / (rec_list[i] + prec_list[i])
        f1_list.append(f1)'''
    logging.warning("RESULT K FOLD")
    logging.warning(results_kfold)

    #logging.warning("\nobtained from ")
    #logging.warning("F1 on each fold " + str(f1_list))
    #logging.warning("precision on each fold " + str(prec_list))
    #logging.warning("recall on each fold " + str(rec_list))
    #logging.warning("accuracy on each fold " + str(acc_list))
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

    upsampleStrategy = False
    downsampleStrategy = False

    text_input_features = False
    exclusion_list = []

    # try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["generate-dataframe=", "input-features=", "dataframe=",
                                                  "true-false-mixed","sampling-strategy=",
                                                  "text-input-features", "error-file"])
    #--generate-dataframe="C:\\fact_checking\\data/dataframe_basic_claimkg.csv", --input-features="C:\\fact_checking\\data\\claimskg_1.0_embeddings_d400_it1000_opdot_softmax_t0.25_trlinear.tsv"
#--dataframe "C:\\fact_checking\\dataframe_CBD_all_11_07_model_6.csv"

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
        if opt == '--sampling-strategy':
            strategy = str(arg)
            if strategy == "upsample":
                upsampleStrategy = True
            if strategy == "downsample":
                downsampleStrategy = False
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

        line = f.readline()
        parts = line.split(sep)
        #lines = f.readline()
        #parts = lines[0].split(sep)
        print("C")
        if text_input_features:
            dims = len(parts) - 1
        else:
            dims = len(parts) - 1
        f.close()
        print("B")
        # create a list of col names
        cnames = ['nodeID']
        for i in range(0, dims - 1):
            cnames.append("feature" + str(i))
        cnames.append('target')

        # Creating vectors for features and class/response Variable
        df = pd.DataFrame(columns=cnames)
        list_of_lists = []
        i_count = 0
        other_count = 0
        print("A")
        f = open(input_path, 'r', encoding='utf8')

        line = f.readline()
        while line:
        #for line in tqdm(lines):
            # print(line.translate(table), end="")
            parts = line.strip().split(sep)
            if parts[0].startswith("<"):
                parts[0] = parts[0][1:-1]
            if not (parts[0].startswith("http://data.gesis.org/claimskg/creative_work/")):
                other_count += 1
                if other_count % 1000000 == 0:
                    print("****read " + str(other_count))
                continue
            parts[1:dims] = [float(part) for part in parts[1:dims]]

            if not text_input_features:
                line_class = get_class(parts[0])
                parts.append(line_class)
            else:
                line_class = parts[-1]

            if true_and_false_vs_mix and (
                    line_class == 'MIXTURE' or line_class == 'TRUE' or line_class == 'FALSE' and parts[0] not in exclusion_list):
                list_of_lists.append(parts)
            elif line_class == 'TRUE' or line_class == 'FALSE' and parts[0] not in exclusion_list:
                list_of_lists.append(parts)

            line = f.readline()
            i_count += 1
            if i_count % 1000 == 0:
                print("read " + str(i_count))

        df = df.append(pd.DataFrame(list_of_lists, columns=df.columns))
        df.to_csv(output_dataframe_path, sep=',')
    else:
        # read file already preprared with feat and target
        df = pd.read_csv(input_dataframe_path, sep='\t', encoding='utf-8')

    print("It is time to balance the dataset")
    # drop claim where claim ID not recognized
    df = df[df.target != -1]
    if true_vs_false:
        df = df[df.target != 2]
    if true_and_false_vs_mix:
        df.loc[df['target'] == 3, 'target'] = 1
    data = df
    x_feature_vectors = data.drop('target', axis=1).drop('nodeID',
                                                         axis=1).values

    # separate minority and majority classes
    false_cls = data[data.target == 1]
    true_cls = data[data.target == 2]

    if false_cls.shape[0] > true_cls.shape[0]:
        minority = true_cls
        majority = false_cls
    else:
        minority = false_cls
        majority = true_cls
    print("shape before sampling")
    print(false_cls.shape[0])
    print(true_cls.shape[0])

    if upsampleStrategy:
        print("upsample strategy adopted")
        # upsample minority
        minority_upsampled = resample(minority,
                                      replace=True,  # sample with replacement
                                      n_samples=len(majority),  # match number in majority class
                                      random_state=27)  # reproducible results

        # combine majority and upsampled minority
        upsampled = pd.concat([majority, minority_upsampled])
        # check new class counts
        print(upsampled.target.value_counts())
        data = upsampled

    elif downsampleStrategy:
        # downsample majority
        majority_downsampled = resample(majority,
                                        replace=False,  # sample without replacement
                                        n_samples=len(minority),  # match minority n
                                        random_state=27)  # reproducible results

        # combine minority and downsampled majority
        downsampled = pd.concat([majority_downsampled, minority])
        # check new class counts
        print("shape after sampling")
        print(downsampled.target.value_counts())
        data = downsampled

    print("check sampling")
    print(data.target.value_counts())
    x_feature_vectors = data.drop('target', axis=1).drop('nodeID', axis=1).values
    print(x_feature_vectors.shape)
    y_class_vector = data['target'].values
    print(y_class_vector.shape)
    print(data['target'].value_counts())

    # test Different Model using KFOLD
    models_dict = dict()
    model_list = specify_models()

    '''my_scores = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro',
    }'''
    my_scores = scoring_functions.overall_scoring()
    #my_scores = {'accuracy': 'accuracy','prec_macro': 'precision_macro','rec_macro': 'recall_macro'}

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
