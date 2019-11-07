import sys

import pandas
from sklearn.model_selection import KFold

seed = 100
if len(sys.argv) > 1:
    seed = int(sys.argv[1])

df = pandas.read_csv("dataframe.csv")

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
fold_index = 1
uris = df['nodeID'].to_list()
for train_index, test_index in kfold.split(df.to_numpy()):
    print("Writing fold " + str(fold_index))
    with open("folds/fold_train_" + str(fold_index), "w", encoding="utf-8") as fold_file:
        for item_index in train_index:
            fold_file.write(uris[item_index] + "\n")
        fold_file.flush()
    with open("folds/fold_test_" + str(fold_index), "w", encoding="utf-8") as fold_file:
        for item_index in test_index:
            fold_file.write(uris[item_index] + "\n")
        fold_file.flush()
    fold_index += 1
