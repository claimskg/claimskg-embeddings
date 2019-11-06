from build_extended_graph.CBDBuilder import *
from build_extended_graph.utils_claimKG import *
from SPARQLWrapper import SPARQLWrapper, JSON

sparql_kg = SPARQLWrapper("https://data.gesis.org/claimskg/sparql")
sparq_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")
sparq_dbpedia = SPARQLWrapper("http://live.dbpedia.org/sparql")

sparq_dbpedia.setTimeout(3600)


if __name__ == '__main__':

    #LOAD THE basic CLAIMsKG
    #for each claim, it loads the author, dbpedia mentions, keyword, publication date, sameAs relations
    basic_g = Graph()
    claim_id_set = set()
    reviewRatingList =  ["1", "2", "3"] #FALSE, MIXTURE, TRUE

    for reviewRating in reviewRatingList:
        basic_g, claim_id_set= get_author_triples_from_claimKG(reviewRating, basic_g, claim_id_set)
        basic_g, claim_id_set = get_mentions_triples_from_claimKG(reviewRating, basic_g, claim_id_set)
        basic_g, claim_id_set = get_keywords_triples_from_claimKG(reviewRating,basic_g, claim_id_set)
        basic_g, claim_id_set = get_publication_date_triples_from_claimKG(reviewRating,basic_g, claim_id_set)
        basic_g, claim_id_set = get_sameAs_triples_from_claimKG(reviewRating, basic_g,claim_id_set)
        print("New size of basic_g " + str(len(basic_g)))
        print("New size of claim id " + str(len(claim_id_set)))

    #EXTEND THE CLAIM KG graph adding the CBD for each dbpedia mention contained in the basic claimKG graph
    extended_g = basic_g
    print("size of extended graph before adding any CBD " + str(len(extended_g)))

    mentions_set = set()
    for s,p,o in basic_g.triples((None, rdflib.term.URIRef("http://schema.org/mentions"), None)):
        mentions_set.add(o)
    print("\tsize of mention set " + str(len(mentions_set)))

    cont = 0
    mentions_list = list(mentions_set)
    mentions_list.sort()
    for resource_uri in mentions_list:
        cont += 1
        #if cont < 7400:
        #    continue
        print(resource_uri)
        extended_g = get_CBD2(resource_uri, extended_g)

        if cont % 100 == 0:
            print("processed " + str(cont) + " out of " + str(len(mentions_set)))


    print("\tsize of extended graph " + str(len(extended_g)))
    # g.serialize("graph_CBD_all.ttl", format='ttl')
    extended_g.serialize("graph_CBD_all_11_07.rdf", format="application/rdf+xml")

    #filtering predicates
    list_of_predicate_to_remove = ["http://dbpedia.org/ontology/wikiPageWikiLink",
                                   "http://dbpedia.org/ontology/wikiPageDisambiguates",
                                   "http://dbpedia.org/ontology/wikiPageExternalLink",
                                   "http://dbpedia.org/ontology/wikiPageID",
                                   "http://dbpedia.org/ontology/wikiPageRedirects",
                                   "http://dbpedia.org/ontology/wikiPageRevisionID",
                                   "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                                   "http://www.w3.org/2000/01/rdf-schema#comment",
                                   "http://www.w3.org/2000/01/rdf-schema#label",
                                   "http://www.w3.org/2000/01/rdf-schema#seeAlso",
                                   "http://www.w3.org/2002/07/owl#differentFrom",
                                   "http://www.w3.org/2003/01/geo/wgs84_pos#geometry",
                                   "http://www.w3.org/2003/01/geo/wgs84_pos#lat",
                                   "http://www.w3.org/2003/01/geo/wgs84_pos#long",
                                   "http://www.w3.org/ns/prov#wasDerivedFrom", "http://xmlns.com/foaf/0.1/depiction",
                                   "http://xmlns.com/foaf/0.1/givenName", "http://xmlns.com/foaf/0.1/homepage",
                                   "http://xmlns.com/foaf/0.1/isPrimaryTopicOf", "http://xmlns.com/foaf/0.1/logo",
                                   "http://xmlns.com/foaf/0.1/nick", "http://xmlns.com/foaf/0.1/page",
                                   "http://dbpedia.org/ontology/abstract", "http://dbpedia.org/ontology/thumbnail"]


    # filtering
    removed_triples = 0

    print("URI " + "\t" + "Number of instances")
    for uri in list_of_predicate_to_remove:
        a = len(list(extended_g.triples((None, rdflib.term.URIRef(uri), None))))
        print(str(uri) + "\t" + str(a))
        removed_triples += (len(extended_g))
        extended_g.remove((None, rdflib.term.URIRef(uri), None))
        removed_triples -= (len(extended_g))

    print("number of removed triples")
    print(removed_triples)

    print("\tsize of extended graph after static filtering " + str(len(extended_g)))
    extended_g.serialize("graph_CBD_with_static_filters_11_07.rdf", format="application/rdf+xml")  # .encode("utf-8")