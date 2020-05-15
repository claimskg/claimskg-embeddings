
import os
from os import path

import numpy
import pandas
from pandas import DataFrame

if not path.exists("./deep/corpus"):
    os.mkdir("./deep/corpus")

df = pandas.read_csv("demandes/new_queries.csv")
print(df)
df = df[["motif", "demande"]]
df["motif"] = "__label__" + df["motif"].astype("str")

num_splits = 10

purposes = pandas.read_csv("motifs/purpose_for_training_upsampled.csv")[["name", "description"]]
purposes.rename(columns={'name': 'motif', 'description': 'demande'}, inplace=True)
purposes["motif"] = "__label__" + purposes["motif"].astype("str")

purposes = purposes.sample(frac=1)

for split in range(num_splits):
    base_path = "./deep/corpus/split_" + str(split)
    if not path.exists(base_path):
        os.mkdir(base_path)

    train, dev, test = numpy.split(df.sample(frac=1), [int(.7 * len(df)), int(.9 * len(df))])  # type: DataFrame

    train = pandas.concat([train, purposes])
    train.to_csv(base_path + "/train.txt", index=False, sep="\t", header=False)
    test.to_csv(base_path + "/test.txt", index=False, sep="\t", header=False)
    dev.to_csv(base_path + "/dev.txt", index=False, sep="\t", header=False)
