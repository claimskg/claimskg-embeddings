from rdflib import Graph
import rdflib
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://data.gesis.org/claimskg/sparql")
sparq_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")


def get_keywords_triples_from_claimKG(ratingValue_, g,claim_id_set):
    # create dict claim uri - keywords
    sparql.setQuery("""    PREFIX schema: <http://schema.org/>
                           PREFIX dbr: <http://dbpedia.org/resource/>
                           SELECT ?claim ?keys WHERE {
                           ?claim a schema:CreativeWork . 
                           ?claim schema:keywords ?keys .     

                           ?claimReview schema:itemReviewed ?claim .                                     
                           ?claimReview schema:reviewRating ?rating .              
                           ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ;
                                schema:ratingValue ?ratingValue FILTER (?ratingValue = """ + str(ratingValue_) + """) . 
                           } """)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # initialize list of lists
    for result in results["results"]["bindings"]:
        claim_id = result["claim"]['value']
        claim_id_set.add(claim_id)
        if "keys" in result:
            keys = result["keys"]['value']
            g.add((rdflib.term.URIRef(claim_id), rdflib.term.URIRef("http://schema.org/keywords"),
                    rdflib.term.Literal(keys)))

    return g,claim_id_set

def get_mentions_triples_from_claimKG(ratingValue_, g,claim_id_set):
    # create dict claim uri - keywords
    sparql.setQuery("""    PREFIX schema: <http://schema.org/>
                           PREFIX dbr: <http://dbpedia.org/resource/>
                           SELECT ?claim ?dbpediaURL WHERE {
                           ?claim a schema:CreativeWork . 
                           ?claim schema:mentions ?mentions . ?mentions <https://www.w3.org/2005/11/its/rdf#taIdentRef> ?dbpediaURL .    

                           ?claimReview schema:itemReviewed ?claim .                                     
                           ?claimReview schema:reviewRating ?rating .              
                           ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ;
                                schema:ratingValue ?ratingValue FILTER (?ratingValue = """ + str(ratingValue_) + """) . 
                           } """)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # initialize list of lists
    for result in results["results"]["bindings"]:
        claim_id = result["claim"]['value']
        claim_id_set.add(claim_id)
        if "dbpediaURL" in result:
            dbpediaURL = result["dbpediaURL"]['value']
            g.add((rdflib.term.URIRef(claim_id), rdflib.term.URIRef("http://schema.org/mentions"),
                    rdflib.term.URIRef(dbpediaURL)))

    return g,claim_id_set


def get_author_triples_from_claimKG(ratingValue_, g,claim_id_set):
    # create dict claim uri - keywords
    sparql.setQuery("""    PREFIX schema: <http://schema.org/>
                           PREFIX dbr: <http://dbpedia.org/resource/>
                           SELECT ?claim ?author WHERE {
                           ?claim a schema:CreativeWork . 
                           ?claim schema:author ?author .   

                           ?claimReview schema:itemReviewed ?claim .                                     
                           ?claimReview schema:reviewRating ?rating .              
                           ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ;
                                schema:ratingValue ?ratingValue FILTER (?ratingValue = """ + str(ratingValue_) + """) . 
                           } """)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # initialize list of lists
    for result in results["results"]["bindings"]:
        claim_id = result["claim"]['value']
        claim_id_set.add(claim_id)
        if "author" in result:
            author = result["author"]['value']
            g.add((rdflib.term.URIRef(claim_id), rdflib.term.URIRef("http://schema.org/author"),
                    rdflib.term.URIRef(author)))

    return g,claim_id_set

def get_publication_date_triples_from_claimKG(ratingValue_, g,claim_id_set):
    # create dict claim uri - keywords
    sparql.setQuery("""    PREFIX schema: <http://schema.org/>
                           PREFIX dbr: <http://dbpedia.org/resource/>
                           SELECT ?claim ?date WHERE {
                           ?claim a schema:CreativeWork . 
                           ?claim schema:datePublished ?date .   

                           ?claimReview schema:itemReviewed ?claim .                                     
                           ?claimReview schema:reviewRating ?rating .              
                           ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ;
                                schema:ratingValue ?ratingValue FILTER (?ratingValue = """ + str(ratingValue_) + """) . 
                           } """)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # initialize list of lists
    for result in results["results"]["bindings"]:
        claim_id = result["claim"]['value']
        claim_id_set.add(claim_id)
        if "date" in result:
            date = result["date"]['value']
            g.add((rdflib.term.URIRef(claim_id), rdflib.term.URIRef("http://schema.org/datePublished"),
                    rdflib.term.Literal(date)))

    return g,claim_id_set

def get_sameAs_triples_from_claimKG(ratingValue_, g,claim_id_set):
    # create dict claim uri - keywords
    sparql.setQuery("""    PREFIX schema: <http://schema.org/>
                           PREFIX dbr: <http://dbpedia.org/resource/>
                           SELECT ?claim ?claim1 WHERE {
                           ?claim a schema:CreativeWork . 
                           ?claim <http://www.w3.org/TR/2004/REC-owl-semantics-20040210/#owl_sameAs> ?claim1 .   

                           ?claimReview schema:itemReviewed ?claim .                                     
                           ?claimReview schema:reviewRating ?rating .              
                           ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ;
                                schema:ratingValue ?ratingValue FILTER (?ratingValue = """ + str(ratingValue_) + """) . 
                           } """)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # initialize list of lists
    for result in results["results"]["bindings"]:
        claim_id = result["claim"]['value']
        claim_id_set.add(claim_id)
        if "claim1" in result:
            claim1 = result["claim1"]['value']
            g.add((rdflib.term.URIRef(claim_id), rdflib.term.URIRef("http://www.w3.org/TR/2004/REC-owl-semantics-20040210/#owl_sameAs"),
                    rdflib.term.URIRef(claim1)))

    return g,claim_id_set

if __name__ == '__main__':
    # create dataframe claim id - true\false - text - author - date
    #WE ARE NOT CURRENTLY CONSIDERING CLAIMS FROM SNOPES ***author is not the correct ones

    #get_true_false_KGgraph_complete()
    exit()

    df = get_true_and_false_claims_and_context()
    #create dict claim id - keywords
    data_keywords = get_true_false_claims_and_keywords()
    #create dict claim id - string mentions + dict string mention - DBpediaURI
    data_mentions, mentions_url_data = get_true_false_claims_and_mentions()