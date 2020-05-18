import traceback

from SPARQLWrapper import SPARQLWrapper

from cls_task.sparql_offset_fetcher import SparQLOffsetFetcher


def get_all_claims(sparql_wrapper: SPARQLWrapper, labels):
    prefixes = """
PREFIX itsrdf: <https://www.w3.org/2005/11/its/rdf#>
PREFIX schema: <http://schema.org/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
    """
    where_body = """
    ?claim a schema:CreativeWork ; schema:text ?claim_text .
  ?claimReview schema:itemReviewed ?claim ; schema:reviewRating ?rating ; schema:headline ?headline.
  ?rating schema:author <http://data.gesis.org/claimskg/organization/claimskg> ; schema:alternateName ?ratingName 
    """

    fetcher = SparQLOffsetFetcher(sparql_wrapper, 10000, where_body=where_body,
                                  select_columns="?claim COALESCE(?claim_text, ?headline) as ?text ?ratingName",
                                  prefixes=prefixes)

    results = fetcher.fetch_all()

    # print(str_query)

    try:
        claims = dict()
        for result in results:
            # check if ?o is a blank node (if yes, then iterate on it -- add on queue )
            id = result["claim"]["value"]
            rating_ = result["ratingName"]['value']
            text_ = str(result["text"]['value']).strip()
            if len(text_) > 0 and rating_ in labels:
                claims[id] = (text_.replace("\n", " ").replace("\t", "")[:511], rating_)
    except:
        print("Exception")
        print(traceback.format_exc())
        return (None, "NOT FOUND")
    return claims
