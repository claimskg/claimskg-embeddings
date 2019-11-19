import getopt
import logging
import logging.config
# from utils import get_class_labels
import sys

import pandas
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from tqdm import tqdm

from cls_task import scoring_functions
from cls_task.split_generation import generate_splits

logging.basicConfig(filename='app.log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

logging.warning("----------New run-------------")

ratings_dict = dict()
with open("ratings.tsv", "r") as ratings:
    for line in ratings.readlines():
        parts = line.strip().split("\t")
        ratings_dict[parts[0]] = parts[1]


def get_class_offline(claimID):
    return ratings_dict[claimID]


def specify_models():
    nbayes = {'name': 'Naive Bayes', 'class': GaussianNB(), 'parameters': {}, 'tentative_best_parameters': {}}

    knear = {
        'name': 'K Nearest Neighbors Classifier',
        'class': KNeighborsClassifier(),
        'parameters': {
            'n_neighbors': range(1, 12)
        },
        'tentative_best_parameters': {
            'n_neighbors': 1
        }
    }

    loglas = {
        'name': "Logistic Regression with LASSO",
        'class': LogisticRegression(penalty='l1'),
        'parameters': {
            'C': [0.1, 1, 10, 100]
        },
        'tentative_best_parameters': {
            'C': 1
        }
    }

    sgdc = {
        'name': "Stochastic Gradient Descent Classifier",
        'class': SGDClassifier(),
        'parameters': {
            'max_iter': [100, 500],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        },
        'tentative_best_parameters': {
            'max_iter': 100,
            'alpha': 0.001
        }

    }

    decis_tree = {
        'name': "Decision Tree Classifier",
        'class': DecisionTreeClassifier(),
        'parameters': {
            'max_depth': range(3, 15)
        },
        'tentative_best_parameters': {
            'max_depth': 14
        }
    }

    ranfor = {
        'name': "Random Forest Classifier",
        'class': RandomForestClassifier(),
        'parameters': {
            'n_estimators': [10, 20, 50, 100, 200]
        },
        'tentative_best_parameters': {
            'n_estimators': 200
        }
    }

    extrerantree = {
        'name': "Extremely Randomized Trees Classifier",
        'class': ExtraTreesClassifier(),
        'parameters': {
            'n_estimators': [10, 20, 50, 100, 200]
        },
        'tentative_best_parameters': {
            'n_estimators': 100
        }
    }
    svc_linear = {
        'name': 'Support Vector Classifier with Linear Kernel',
        'class': LinearSVC(),
        'parameters': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'tentative_best_parameters': {
            'C': 0.001
        }
    }

    sv_radial = {
        'name': 'Support Vector Classifier with Radial Kernel',
        'class': SVC(kernel='rbf'),
        'parameters': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'tentative_best_parameters': {
            'C': 1,
            'gamma': 1
        }
    }

    nn_mlp = {
        'name': 'multi-layer perceptron',
        'class': MLPClassifier(solver='adam', learning_rate="adaptive", alpha=1e-5, random_state=100,
                               hidden_layer_sizes=(300, 10)),
        'parameters': {
        }
    }
    lis = list([
        extrerantree, ranfor, nbayes, sv_radial, loglas, sgdc, decis_tree, knear
        # nn_mlp
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


def make_cls(model_dict, X, y, splits, jobs=1, metric='f1', estimate_parameters=False):
    '''
		model_dict : We will pass in the dictionaries from the list you just created one by one to this parameter
		X: The input data
		y: The target variable
		metric : The name of a metric to use for evluating performance during cross validation. Please give this parameter a default value of 'f1' which is the F measure.
		k : The number of folds to use with cross validation, the default should be 5
		'''

    name = model_dict['name']
    param_grid = model_dict['parameters']
    cls_app = model_dict['class']
    logging.warning('MODEL ' + str(name))
    if len(param_grid) > 0 and estimate_parameters:
        print("Grid Search for " + str(model_dict['name']))
        grid_obj = GridSearchCV(model_dict['class'],
                                param_grid,
                                scoring=metric,
                                cv=splits,
                                refit='accuracy', n_jobs=jobs)
        grid_obj = grid_obj.fit(X, y)
        best_parameters = grid_obj.best_params_
        logging.warning(best_parameters)

        best_score = grid_obj.best_score_
        # best_score= clf.fit(X,y).best_score_
        best_model = grid_obj
        cls_app.set_params(**best_parameters)
    elif not estimate_parameters:
        tentative_best_parameters = model_dict['tentative_best_parameters']
        cls_app.set_params(**tentative_best_parameters)
    # scoring = ['precision_macro', 'recall_macro', 'accuracy']

    # HERE ADD SOURCE CODE WITH NEW SCORING FUNCTION
    print("Cross Validation for " + str(model_dict['name']))
    scoring = scoring_functions.overall_scoring()
    results_kfold = cross_validate(cls_app, X, y, scoring=scoring, cv=splits, return_train_score=False, n_jobs=jobs)

    # HERE new str_out for logging
    str_out = str(name) + "\t"
    acc_list = list(results_kfold['test_accuracy'])
    str_out += str(sum(acc_list) / len(acc_list)) + "\t"

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

    tp_list = list(results_kfold['test_tp'])
    str_out += str(sum(tp_list) / len(tp_list)) + "\t"
    tn_list = list(results_kfold['test_tn'])
    str_out += str(sum(tn_list) / len(tn_list)) + "\t"

    fp_list = list(results_kfold['test_fp'])
    str_out += str(sum(fp_list) / len(fp_list)) + "\t"
    fn_list = list(results_kfold['test_fn'])
    str_out += str(sum(fn_list) / len(fn_list)) + "\t"

    # logging.warning("+++++++++++++++++++++")
    logging.warning(str_out)
    logging.warning("+++++++++++++++++++++")

    if len(param_grid) == 0 or not estimate_parameters:
        best_score = sum(acc_list) / len(acc_list)
        best_model = name

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

    write_splits = False

    estimate_parameters = False

    jobs = 1

    strategy = "None"

    # try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["input-features=",
                                                  "true-false-mixed", "sampling-strategy=", "rating-path=",
                                                  "text-input-features", "error-file", "write-splits", "jobs=",
                                                  "estimate-parameters"])
    # --generate-dataframe="C:\\fact_checking\\data/dataframe_basic_claimkg.csv", --input-features="C:\\fact_checking\\data\\claimskg_1.0_embeddings_d400_it1000_opdot_softmax_t0.25_trlinear.tsv"
    # --dataframe "C:\\fact_checking\\dataframe_CBD_all_11_07_model_6.csv"

    for opt, arg in opts:

        if opt == '--input-features':
            input_path = str(arg)
            # HERE add log
            logging.warning("--input-features\t" + input_path)
        if opt == '--exclusion-file':
            exclusion_file_path = str(arg)
            exclusion_file = open(exclusion_file_path, "r", encoding="utf8")
            # HERE add log
            logging.warning("--exclusion-file\t" + exclusion_file_path)
            exclusion_list = exclusion_file.readlines()
        if opt == '--text-input-features':
            text_input_features = True
            # HERE add log
            logging.warning("--text-input-features\t" + str(text_input_features))
        if opt == '--true-false-mixed':
            true_vs_false = False
            true_and_false_vs_mix = True
            # HERE add log
            logging.warning("--cls problem\t true and false VS mixture")
        if opt == '--sampling-strategy':
            strategy = str(arg)
            # HERE change var value + log
            if strategy == "upsample":
                upsampleStrategy = True
                logging.warning("--sampling-strategy\t" + strategy)
            if strategy == "downsample":
                downsampleStrategy = True
                logging.warning("--sampling-strategy\t" + strategy)
        if opt == "--write-splits":
            write_splits = True
        if opt == "--estimate-parameters":
            estimate_parameters = True
        if opt == "--jobs":
            jobs = int(arg)
    # HERE add logs
    if true_vs_false:
        logging.warning("--cls problem\t true VS false")
    if strategy != "upsample" and strategy != "downsample":
        logging.warning("--sampling-strategy\tNONE")
    logging.warning("")
    # except:
    #     print('Arguments parser error, try -h')
    #     exit()

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
    f.close()

    # create a list of col names
    cnames = ['nodeID']
    # HERE remove -1 from range
    for i in range(0, dims):
        cnames.append("feature" + str(i))
    cnames.append('target')

    # Creating vectors for features and class/response Variable
    list_of_lists = []
    i_count = 0
    other_count = 0

    for line in tqdm(lines):
        # print(line.translate(table), end="")
        parts = line.strip().split(sep)
        if parts[0].startswith("<"):
            parts[0] = parts[0][1:-1]
        if not (parts[0].startswith("http://data.gesis.org/claimskg/creative_work/")):
            continue
        parts[1:dims] = [float(part) for part in parts[1:dims]]

        if not text_input_features:
            line_class = get_class_offline(parts[0]).strip()
            parts.append(line_class)
        else:
            line_class = parts[-1]

        if true_and_false_vs_mix and (
                line_class == 'MIXTURE' or line_class == 'TRUE' or line_class == 'FALSE' and parts[
            0] not in exclusion_list):
            list_of_lists.append(parts)
        elif line_class == 'TRUE' or line_class == 'FALSE' and parts[0] not in exclusion_list:
            list_of_lists.append(parts)

    df = pandas.DataFrame(list_of_lists, columns=cnames)

    print("It is time to balance the dataset")
    # drop claim where are not interested in
    df = df[df.target != 'OTHER']  # -1
    if true_vs_false:
        df = df[df.target != 'MIXTURE']  # 2
    if true_and_false_vs_mix:
        df.loc[df['target'] == 'TRUE', 'target'] = 'FALSE'  # df.loc[df['target'] == 3, 'target'] = 1
    data = df
    x_feature_vectors = data.drop('target', axis=1).drop('nodeID',
                                                         axis=1).values
    # HERE add code
    # separate minority and majority classes
    if true_vs_false:
        false_cls = data[data.target == 'FALSE']  # 1
        true_cls = data[data.target == 'TRUE']  # 3
    if true_and_false_vs_mix:
        false_cls = data[data.target == 'FALSE']  # 1
        true_cls = data[data.target == 'MIXTURE']  # 2

    if false_cls.shape[0] > true_cls.shape[0]:
        minority = true_cls
        majority = false_cls
    else:
        minority = false_cls
        majority = true_cls

    if upsampleStrategy:
        # print("upsample strategy adopted")
        # upsample minority
        minority_upsampled = resample(minority,
                                      replace=True,  # sample with replacement
                                      n_samples=len(majority),  # match number in majority class
                                      random_state=27)  # reproducible results

        # combine majority and upsampled minority
        upsampled = pandas.concat([majority, minority_upsampled])
        # check new class counts
        # print(upsampled.target.value_counts())
        data = upsampled

    elif downsampleStrategy:
        # downsample majority
        majority_downsampled = resample(majority,
                                        replace=False,  # sample without replacement
                                        n_samples=len(minority),  # match minority n
                                        random_state=27)  # reproducible results

        # combine minority and downsampled majority
        downsampled = pandas.concat([majority_downsampled, minority])
        # check new class counts
        # print("shape after sampling")
        # print(downsampled.target.value_counts())
        data = downsampled

    # HERE replace target value
    data.loc[data['target'] == 'TRUE', 'target'] = 3
    data.loc[data['target'] == 'MIXTURE', 'target'] = 2
    data.loc[data['target'] == 'FALSE', 'target'] = 1

    # print("check sampling")
    # print(data.target.value_counts())
    x_feature_vectors = data.drop('target', axis=1).drop('nodeID', axis=1).values
    # print(x_feature_vectors.shape)
    y_class_vector = data['target'].values

    # HERE add type specification
    y_class_vector = y_class_vector.astype("int")
    print(y_class_vector.shape)
    print(data['target'].value_counts())

    # test Different Model using KFOLD
    models_dict = dict()
    model_list = specify_models()

    # HERE more fine grained scoring output
    my_scores = scoring_functions.overall_scoring()

    splits, kfold = generate_splits(data, seed=100, write=write_splits)

    if not write_splits:
        for model_dict in tqdm(model_list):
            print(make_cls(model_dict,
                           x_feature_vectors,
                           y_class_vector,
                           kfold,
                           jobs=jobs,
                           metric=my_scores, estimate_parameters=estimate_parameters))
