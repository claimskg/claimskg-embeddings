# claimskg-embeddings
This repository contains all the code and data to reproduce the experiments in the paper entitled "Claim Embeddings: an Empirical Study" on the basis of the ClaimsKG knowledge graph of fact-checked claims. 



All the graphs and embeddings have been generated on the basis of ClaimsKG v1 ( https://zenodo.org/record/2628745), but all the steps can be reproduced with the current version (v2) that is online in the official SPARQL endpoint  (<https://data.gesis.org/claimskg/sparql>).



**The writing for this page is in progress.**



# 1. Generating the graphs

The graphs are not generated from the full ClaimsKG graph, but from a subset that only keeps claims and associated metadata (all claim reviews and related metadata are removed). To generate the graphs, you need to have a functioning SPARQL endpoint of ClaimsKG. You can use virtuoso-docker to easily set-up an endpoint and load ClaimsKG.

You can use the `build_extended_graph/generate_graph.py ` script. 

_Requirements_: SPARQLWrapper, tqdm, redis (a running redis server to cache sparql queries)

Syntax: 

```shell
python build_extended_graph/generate_graph.py [SPARQL URI | LOCAL_FILE] [extend]
```

If you just provide the sparql endpoint URI, the graph for `ClaimsKG_B` will be created (`ClaimsKG_B.rdf`), if you add `extend` as a second argument, then the extended `ClaimsKG_E` graph will be created (`ClaimsKG_E.rdf`).



You can find a list of predicates imported from DBPedia in the extended `ClaimsKG_E` graph as of December, 12th 2019 can be found here: [predicate CBD.pdf](predicate CBD.pdf).

# 2. Computing the Embeddings

## 2.1 Graph Embeddings

*Prerequisite*: You will need to install PytorchBigGraph on your python distribution. You may do so with `pip install torchbiggraph`. For more complete instructions, please refer to the github repository: [facebookresearch/PyTorch-BigGraph](<https://github.com/facebookresearch/PyTorch-BigGraph>).

### 2.1.1 Preparing the graph files for Pytorch BigGraph

### 2.1.2 Generating the HFDS files 

### 2.1.3  Running the grid search for the graph embeddings

### 2.1.4 Manually training the embeddings 

### 2.1.5 Extracting graph embeddings as TSV

## 2.2 Text Embeddings

# 3. Pre-trained Embeddings from the Paper

You may find all the pre-trained embeddings used in the experiments here: https://bit.ly/36qGvHr

The directory structure is the following:

- graph
  - ClaimsKG-B.tsv.bz2
  - ClaimsKG_E.tsv.bz2
- text
  - Claims_Text_Glove.csv.bz2
  - Claims_Text_XLM.csv.bz2

You will need to decompress the files before using them to reproduce the experiments. 



# 4. Reproducing the Fact verification results

The full results tables from the paper are available here: 

- [Unbalanced Dataset Results](complete-results/ClaimsKG Embeddings Results - Unbalanced data.pdf)

- [Upsampled Dataset Results](complete-results/ClaimsKG Embeddings Results - upsampled data.pdf)











