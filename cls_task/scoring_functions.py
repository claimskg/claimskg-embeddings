from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]


def overall_scoring():
    scoring = {'accuracy': 'accuracy',
               'f1': 'f1',
               'precision': 'precision',
               'recall':'recall',
               'f1_macro': 'f1_macro',
               'precision_macro': 'precision_macro',
               'recall_macro': 'recall_macro',
               'f1_micro': 'f1_micro',
               'precision_micro': 'precision_micro',
               'recall_micro': 'recall_micro',
               'f1_weighted': 'f1_weighted',
               'precision_weighted': 'precision_weighted',
               'recall_weighted': 'recall_weighted',
               'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn)}

    return scoring
