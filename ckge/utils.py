from SPARQLWrapper import SPARQLWrapper

from ckge.sparql_offset_fetcher import SparQLOffsetFetcher


def get_all_claims(sparql_wrapper: SPARQLWrapper, labels):
    prefixes = """
PREFIX itsrdf: <https://www.w3.org/2005/11/its/rdf#>
PREFIX schema: <http://schema.org/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
    """
    where_body = """
?claim schema:text ?claim_text_l; 
         schema:keywords ?kwe; 
         schema:author ?author_ent. 

  ?review schema:itemReviewed ?claim;
          schema:reviewRating ?rating;
          schema:headline ?headline_l.

  ?author_ent schema:name ?author_name_l.
  
  ?kwe schema:name ?kw_l.
  
?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ; schema:alternateName ?ratingName.
BIND(str(?author_name_l) as ?author_name)
BIND(str(?kw_l) as ?kw)
BIND(str(?claim_text_l) as ?claim_text)
BIND(str(?headline_l) as ?headline)
    """

    fetcher = SparQLOffsetFetcher(sparql_wrapper, 10000, where_body=where_body,
                                  select_columns="distinct ?claim COALESCE(?claim_text, ?headline) as ?text ?ratingName ?kw ?author_name",
                                  prefixes=prefixes)

    results = fetcher.fetch_all()

    # print(str_query)

    claims = dict()
    ids = set()
    ratings = {}
    texts = {}
    keywords = {}
    authors = {}
    for result in results:
        # check if ?o is a blank node (if yes, then iterate on it -- add on queue )
        id = result["claim"]["value"]
        ids.add(id)
        ratings[id] = result["ratingName"]['value']
        texts[id] = str(result["text"]['value']).strip().replace("\n", " ").replace("\t", "")[:511]
        if id not in keywords.keys():
            keywords[id] = set()
        keywords[id].add(str(result["kw"]["value"]))
        authors[id] = str(result["author_name"]["value"])

    for id in ids:
        author = authors[id]
        rating = ratings[id]
        kws = " ".join(keywords[id])
        text = texts[id]
        if rating in labels:
            claims[id] = (text, rating, kws, author)

    return claims
