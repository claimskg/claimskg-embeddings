import csv
import sys
import random
import os
from pathlib import Path
import traceback

# from utils import get_class_labels
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph
from rdflib.namespace import RDF, FOAF
import rdflib
import urllib
import logging.config

import flair, torch
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

from flair.embeddings import BertEmbeddings
from flair.embeddings import OpenAIGPTEmbeddings
from flair.embeddings import OpenAIGPT2Embeddings
from flair.embeddings import TransformerXLEmbeddings
from flair.embeddings import XLNetEmbeddings
from flair.embeddings import XLMEmbeddings
from flair.embeddings import RoBERTaEmbeddings
from flair.embeddings import ELMoEmbeddings

# python cls_embed_claim_from_text.py ../../../data/entities.list ../../../data/claimskg.dataset.csv ../../../data/data_embeddings_utils/text_embeddings_claims

sparql_kg = SPARQLWrapper("http://localhost:8890/sparql")

def get_info(claimID):

    str_query = """
    PREFIX schema: <http://schema.org/>
    PREFIX nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>

        SELECT ?text ?ratingName WHERE {

        <""" + claimID + """> schema:text ?text .

        ?claimReview schema:itemReviewed <""" + claimID + """> ; schema:reviewRating ?rating .
        ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ; schema:alternateName ?ratingName }
    """

    #print(str_query)
    
    try:
        #print(str_query)
        sparql_kg.setQuery(str_query)  # 1 false, 3 true
        sparql_kg.setReturnFormat(JSON)
        results = sparql_kg.query().convert()
        rating_ = str(-1)
        text_ = "NOT FOUND"
		
        for result in results["results"]["bindings"]:
            # check if ?o is a blank node (if yes, then iterate on it -- add on queue )
            rating_ = result["ratingName"]['value']
            text_ = str(result["text"]['value'])
    except:
        print("Exception -- Skipped claim ID " + str(claimID))
        print(traceback.format_exc())
        return (None, "NOT FOUND")
    return (rating_,text_)

def buildDataset(uri_list_file, file_path):
    '''
    Generate a file storing all descriptions for all URIs
    '''
    print("generating data set: ",file_path)

    with open(uri_list_file) as uri_file, open(file_path, mode='w') as f:

        csv_writter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writter.writerow(['URI', 'CLAIM_TEXT', 'CLASS'])

        count = 0
        count_info_not_found = 0

        for uri_l in uri_file:
            uri = uri_l.strip()
            if len(uri) == 0: continue
            print(uri)
            class_, text_ = get_info(uri)
            if class_ is None:
                class_ = -1
                count_info_not_found += 1

            csv_writter.writerow([uri, text_, class_])
            count += 1
        print("count: ",str(count))
        print("count_info_not_found: ",str(count_info_not_found))

    print("data set generated")


def buildEmbeddings(dataset_fp,emb_dir,configs):

    flair.device = torch.device('cpu') 

    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
        print("creating dir", emb_dir)

    '''
        configs = {
            "simple" : simple_conf,
            "advanced" : conf_trans
        }
    '''

    nb_emb_strats = len(configs["simple"]) + len(configs["advanced"])
    strat_processed = 1

    for conf in configs["simple"]:
        print(str(strat_processed),"/",str(nb_emb_strats),conf)
        embeddings = WordEmbeddings(conf) 
        document_embeddings = DocumentPoolEmbeddings([embeddings])
        emb_file = os.path.join(emb_dir, conf + ".csv")
        buildEmbedding(dataset_fp, emb_file,document_embeddings)
        strat_processed +=1

    for conf in configs["advanced"]:
        print(str(strat_processed),"/",str(nb_emb_strats),conf)
        pkg  = "flair.embeddings"
        name = conf
        embeddings = getattr(sys.modules[pkg], name)() # calling constructor defined in conf
        document_embeddings = DocumentPoolEmbeddings([embeddings])
        emb_file = os.path.join(emb_dir, conf + ".csv")
        buildEmbedding(dataset_fp, emb_file,document_embeddings)
        strat_processed +=1




            


def buildEmbedding(dataset_fp,emb_file,document_embeddings):

    print("Building embeddings: ",emb_file)

    file_error = emb_file+'.error'

    # Expect Header 'URI', 'DESCRIPTION', 'CLASS'
    with open(dataset_fp) as f_read, open(emb_file,'w') as f_write, open(file_error,'w') as f_write_error:

        csv_reader = csv.reader(f_read, delimiter=',')
        csv_writer = csv.writer(f_write, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer_error = csv.writer(f_write_error, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        line_count = 0
        error_count = 0
        cids = {}
        
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
                for i in range(0,len(row)):
                    cids[row[i]] = i
            else:

                if line_count % 100 == 0: print ("processing ",str(line_count), "errors:",error_count, end="\r")

                uri    = row[cids['URI']]
                claim  = row[cids['CLAIM_TEXT']]
                class_ = row[cids['CLASS']]

                try:

                    if uri == 'http://data.gesis.org/claimskg/creative_work/19d1fb43-aa02-5820-9559-5643872230bc':
                        continue

                    s_claim = Sentence(claim)
                    document_embeddings.embed(s_claim)
                    s_claim_emb = s_claim.get_embedding()
                    s_claim_row = s_claim_emb.cpu().detach().numpy().tolist()

                    s_claim_row.insert(0,uri)  # Add URI in first element
                    s_claim_row.append(class_) # Add class in last element
                    csv_writer.writerow(s_claim_row)

                except:
                    print("Exception -- Skipped claim ID " + str(uri))
                    print(traceback.format_exc())
                    error_count += 1
                    csv_writer_error.writerow([uri])

                line_count += 1
        
        print(f' {line_count - 1} embeddings saved at {emb_file}')
        print(f' {error_count} errors {file_error}')


def getAllConfigs():

    print("building configurations")
    configs = []

    ############################################################################################################
    # Transformer embeddings
    # https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md
    # + Elmo + Fastext (see en in simple conf)
    ############################################################################################################
    conf_trans = []

    conf_trans = [
        "BertEmbeddings",
        "OpenAIGPTEmbeddings",
        "OpenAIGPT2Embeddings",
        "TransformerXLEmbeddings",
        "XLNetEmbeddings",
        "XLMEmbeddings",
        "RoBERTaEmbeddings",
        "ELMoEmbeddings"
    ]

    simple_conf = [
        "en-glove",
        "en-extvec",
        "en-crawl",
        "en-twitter",
        "en-turian",
        "en" # FastText
    ]
    
    configs = {
        "simple" : simple_conf,
        "advanced" : conf_trans
    }

    return configs



if __name__ == "__main__":

    if len(sys.argv) < 4:
        print('[0] URI list (one URI per line)')
        print('[1] output dataset file path')
        print('[2] embedding directory')
        exit()

    # Generate the list of URIs
    # (base) seb@opencuda:/media/Baie-MD1400/data/claim_kg$ cat data/pytorch_big_graph/graph_CBD_with_static_filters_11_07/data/entities.dict | grep claimskg | awk '{print $2}' | awk '{print substr($0, 2, length($0) - 2)}' > data/entities.list


    uri_list_file = sys.argv[1]
    dataset_fp = sys.argv[2]
    emb_dir = sys.argv[3]
    
    print("Running Embeddings builder")
    if not os.path.exists(dataset_fp):
        buildDataset(uri_list_file, dataset_fp)
    else: 
        print('skipping dataset generation')
    
    # List of all word embeddings
    # https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md

    configs = getAllConfigs()
    
    buildEmbeddings(dataset_fp,emb_dir,configs)
