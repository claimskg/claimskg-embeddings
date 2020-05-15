import os
from typing import List

import pandas
from pandas import DataFrame

directory = os.fsencode("annotations/")

annotator_dataframes = []  # type: List[DataFrame]

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        annotator_dataframes.append(pandas.read_csv(os.path.join(directory.decode("utf-8"), filename)))

target_dataframe = annotator_dataframes[0].copy()
class_column_labels = ["education", "healthcare", "immigration", "environment", "taxes", "elections", "crime"]
for index, target_row in target_dataframe.iterrows():
    class_histogram = {}

    for class_label in class_column_labels:
        class_histogram[class_label] = 0

    for ann_index in range(len(annotator_dataframes)):
        adf = annotator_dataframes[ann_index]
        arow = adf.iloc[index,]
        for class_label in class_column_labels:
            annotation = arow[class_label]
            if isinstance(annotation, str) and len(annotation) > 0:
                class_histogram[class_label] += 1

    for class_label in class_column_labels:
        count = class_histogram[class_label]
        if count > 2:
            target_dataframe.at[index, class_label] = 1
        else:
            target_dataframe.at[index, class_label] = 0

target_dataframe.to_csv("gold_annotations.csv", index=False)
