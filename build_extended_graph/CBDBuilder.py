import json

import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON
from redis import StrictRedis

sparql_kg = SPARQLWrapper("https://data.gesis.org/claimskg/sparql")
sparq_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")
redis = StrictRedis()


def get_CBD2(resourceURI, g):
    queue = list()
    visited = set()
    queue.append(rdflib.term.URIRef(resourceURI))

    while len(queue) > 0:
        elemURI = queue.pop(0)
        str_elemURI = str(elemURI)
        if "http://" in str(elemURI) or "nodeID" in str(elemURI):
            str_elemURI = "<" + str_elemURI + ">"
        sparq_dbpedia.setQuery(""" SELECT ?p ?o WHERE {""" + str_elemURI + """ ?p ?o . }""")  # 1 false, 3 true
        sparq_dbpedia.setReturnFormat(JSON)
        results = redis.get("____dbpedia____" + str_elemURI)
        if results is None:
            results = sparq_dbpedia.query().convert()
            results = json.dumps(results)
            redis.set("____dbpedia____" + str_elemURI, results)
        results = json.loads(results)

        # initialize list of lists
        res_setOfO = ""
        bnode_setOfO = ""

        for result in results["results"]["bindings"]:
            # check if ?o is a blank node (if yes, then iterate on it -- add on queue )
            if result["o"]['type'] == "bnode":
                g.add((elemURI, rdflib.term.URIRef(result["p"]['value']), rdflib.term.BNode(result["o"]['value'])))
                bnode_setOfO += "<" + str(rdflib.term.BNode(result["o"]['value'])) + ">,"
                res_setOfO += "<" + str(rdflib.term.BNode(result["o"]['value'])) + ">,"
                if result["o"]['value'] not in visited:
                    queue.append(rdflib.term.BNode((result["o"]['value'])))
            # Literal
            if result["o"]['type'] == "literal":
                g.add((elemURI, rdflib.term.URIRef(result["p"]['value']), rdflib.term.Literal(result["o"]['value'])))
            if result["o"]['type'] == "typed-literal":
                g.add((elemURI, rdflib.term.URIRef(result["p"]['value']), rdflib.term.Literal(result["o"]['value'])))
            if result["o"]['type'] == "uri":
                g.add((elemURI, rdflib.term.URIRef(result["p"]['value']), rdflib.term.URIRef(result["o"]['value'])))
                res_setOfO += "<" + str(rdflib.term.URIRef(result["o"]['value'])) + ">,"

        visited.add(elemURI)
        bnode_setOfO = bnode_setOfO[:-1]

        # iterate on blank node
        sparq_dbpedia.setQuery(""" PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
                                   SELECT ?bnode ?p ?u WHERE {
                                          ?bnode ?p ?u 
                                          FILTER (?bnode IN (""" + bnode_setOfO + """))}""")  # 1 false, 3 true

        sparq_dbpedia.setReturnFormat(JSON)

        results = redis.get("_____dbpedia_bnode____" + bnode_setOfO)
        if results is None:
            results = sparq_dbpedia.query().convert()
            results = json.dumps(results)
            redis.set("_____dbpedia_bnode____" + bnode_setOfO, results)
        results = json.loads(results)

        for result in results["results"]["bindings"]:
            if result["u"]['type'] == "bnode":
                g.add((rdflib.term.BNode(result["bnode"]['value']), rdflib.term.URIRef(result["p"]['value']),
                       rdflib.term.BNode(result["u"]['value'])))
                if result["u"]['value'] not in visited:
                    queue.append(rdflib.term.BNode(result["u"]['value']))
            if result["u"]['type'] == "literal":  # Literal
                g.add((rdflib.term.BNode(result["bnode"]['value']), rdflib.term.URIRef(result["p"]['value']),
                       rdflib.term.Literal(result["u"]['value'])))
            if result["u"]['type'] == "typed-literal":
                g.add((rdflib.term.BNode(result["bnode"]['value']), rdflib.term.URIRef(result["p"]['value']),
                       rdflib.term.Literal(result["u"]['value'])))
            if result["u"]['type'] == "uri":
                g.add((rdflib.term.BNode(result["bnode"]['value']), rdflib.term.URIRef(result["p"]['value']),
                       rdflib.term.URIRef(result["u"]['value'])))

            visited.add(rdflib.term.BNode(result["bnode"]['value']))

        # iterate on reification
        sparq_dbpedia.setQuery(""" PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
                                           SELECT ?rnode ?p ?u WHERE {
                                                  ?rnode rdf:type rdf:Statement . 
                                                  ?rnode ?p ?u 
                                                  FILTER (?rnode IN (""" + bnode_setOfO + """))}""")  # 1 false, 3 true

        results = redis.get("_____dbpedia_bnode_reif____" + bnode_setOfO)
        if results is None:
            results = sparq_dbpedia.query().convert()
            results = json.dumps(results)
            redis.set("_____dbpedia_bnode_reif____" + bnode_setOfO, results)
        results = json.loads(results)


        for result in results["results"]["bindings"]:
            if result["u"]['type'] == "bnode":
                g.add((rdflib.term.BNode(result["rnode"]['value'])), rdflib.term.URIRef(result["p"]['value']),
                      rdflib.term.BNode(result["u"]['value']))
                if result["u"]['value'] not in visited:
                    queue.append(rdflib.term.BNode(result["u"]['value']))
            if result["u"]['type'] == "literal":  # Literal
                g.add((rdflib.term.BNode(result["rnode"]['value'])), rdflib.term.URIRef(result["p"]['value']),
                      rdflib.term.Literal(result["u"]['value']))
            if result["u"]['type'] == "typed-literal":
                g.add((rdflib.term.BNode(result["rnode"]['value'])), rdflib.term.URIRef(result["p"]['value']),
                      rdflib.term.Literal(result["u"]['value']))
            if result["u"]['type'] == "uri":
                g.add((rdflib.term.BNode(result["rnode"]['value'])), rdflib.term.URIRef(result["p"]['value']),
                      rdflib.term.URIRef(result["u"]['value']))

            visited.add(rdflib.term.BNode(result["rnode"]['value']))

    return g


if __name__ == '__main__':
    resource_name = "http://dbpedia.org/resource/Apple_Inc."

    g = get_CBD2(resource_name)
    print(len(g))
    for subject, predicate, obj in g:
        print(str(subject) + "\t" + str(predicate) + "\t" + str(obj) + "\t")
