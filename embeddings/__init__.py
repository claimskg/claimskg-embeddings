import json
from enum import Enum
from typing import List

import numpy
from SPARQLWrapper import SPARQLWrapper, JSON
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TokenEmbeddings
from kbc.datasets import Dataset
from kbc.embeddings import KnowledgeGraphEmbeddingExtractor
from kbc.models import KBCModel
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm, trange


class FlairTransformer(BaseEstimator, TransformerMixin):
    """
    a general class for creating a machine learning step in the machine learning pipeline
    """

    def __init__(self, embeddings: List[TokenEmbeddings],
                 fine_tune_mode="linear",
                 pooling: str = "mean", batch_size=32, ):
        """
        constructor
        """
        super(FlairTransformer, self).__init__()
        self.embedder = DocumentPoolEmbeddings(embeddings=embeddings, fine_tune_mode=fine_tune_mode, pooling=pooling)
        self.batch_size = batch_size
        self.vector_cache = {}
        self.dataset_cache = {}

    def fit(self, X, y=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        # No fitting needed, using pre-trained embeddings_baseline
        return self

    def transform(self, X, y=None, **kwargs):
        """
        an abstract method that is used to transform according to what happend in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """

        X = X['text']

        dataset_hash = hash(str(X) + str(self.embedder.__dict__))
        if dataset_hash in self.dataset_cache:
            return self.dataset_cache[dataset_hash]
        else:
            embeddings = []

            for first in trange(0, len(X), self.batch_size):
                subset = X[first:first + self.batch_size]
                sentences = []
                for element in subset:
                    sentence = Sentence(element)
                    # sentence.tokens = sentence.tokens[:200]
                    sentences.append(sentence)

                self.embedder.embed(sentences)
                for sentence in sentences:
                    key = sentence.to_original_text()
                    if key in self.vector_cache.keys():
                        vector = self.vector_cache[key]
                    else:
                        vector = sentence.get_embedding().cpu().detach().numpy()
                        self.vector_cache[key] = vector
                    embeddings.append(vector)

            embedding_dataset = numpy.vstack(embeddings)
            self.dataset_cache[dataset_hash] = embedding_dataset
            return embedding_dataset

    def fit_transform(self, X, y=None, **kwargs):
        """
        perform fit and transform over the data
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        return self.transform(X, y)


class GraphEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dataset: Dataset, model: KBCModel):
        super(GraphEmbeddingTransformer, self).__init__()
        self.embedder = KnowledgeGraphEmbeddingExtractor(dataset, model)

    def fit(self, X, y=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        # No fitting needed, using pre-trained embeddings_baseline
        return self

    def transform(self, X, y=None, **kwargs):
        """
        an abstract method that is used to transform according to what happend in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """

        X = X['claim']

        embeddings = []
        for element in tqdm(X.tolist()):
            claim_embedding = self.embedder.left_hand_side_entity_embedding(element).numpy()
            embeddings.append(claim_embedding)

        embedding_dataset = numpy.vstack(embeddings)
        return embedding_dataset

    def fit_transform(self, X, y=None, **kwargs):
        """
        perform fit and transform over the data
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        return self.transform(X, y)


class NeighbourhoodVectorConcatStrategy(Enum):
    CONCAT_ALL = 0,
    CONCAT_TRIPLES = 1,


class ClamsKGGraphEmbeddingTransformer(GraphEmbeddingTransformer):
    def __init__(self, dataset: Dataset, model: KBCModel, sparql_endpoint: str,
                 concat_strategy: NeighbourhoodVectorConcatStrategy = NeighbourhoodVectorConcatStrategy.CONCAT_ALL,
                 bidirectional: bool = False):
        """

        :param dataset: kbc Dataset object, where the graph is loaded
        :param model: kbc Model class where the trained graph embedding model is loaded
        :param sparql_endpoint: Sparql endpoint serving the same graph as the one used to train the graph embeddings
        :param concat_strategy: Concatenation strategy for neighbourhood vectors.
            - "concat_all" Concatenate and zero-pad all individual neighbourhood vectors
            - "concat_triples" First compute triple vectors (LHSxRELxRHS) and then concatenate them
        """
        super(ClamsKGGraphEmbeddingTransformer, self).__init__(dataset, model)
        self.sparql_endpoint = sparql_endpoint
        self.concat_strategy = concat_strategy
        self.bidirectional = bidirectional

    def fit(self, X, y=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        # No fitting needed, using pre-trained embeddings_baseline
        return self

    def transform(self, X, y=None, **kwargs):
        """
        an abstract method that is used to transform according to what happend in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """

        X = X['claim']

        wrapper = SPARQLWrapper(self.sparql_endpoint)

        embeddings = []
        for element in tqdm(X.tolist()):
            sparql_query = f"""
                PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
                PREFIX thesoz: <http://lod.gesis.org/thesoz/>
                PREFIX unesco: <http://vocabularies.unesco.org/thesaurus/>
                PREFIX schema: <http://schema.org/>
                PREFIX dct: <http://purl.org/dc/terms/>
    
                SELECT * WHERE {{
                  ?review schema:itemReviewed <{element}>. 
                  ?review schema:mentions ?rentity.
    
                  <{element}> schema:author ?author;
                       schema:datePublished ?date;
                       schema:keywords ?kw.
    
                  OPTIONAL {{
                      <{element}> schema:mentions ?centity. 
                  }}      
                }}
                """

            wrapper.setQuery(sparql_query)
            wrapper.setReturnFormat(JSON)
            result = wrapper.query().response.read()

            response_string = str(result, 'utf-8')
            response = json.loads(response_string)

            review_entities = set()
            claim_entities = set()
            review = None
            claim = element
            date = None
            author = None
            keywords = set()

            to_stack = []

            for binding in response['results']['bindings']:
                if review is None:
                    review = binding['review']['value']
                review_entities.add(binding['rentity']['value'])
                if author is None:
                    author = binding['author']['value']
                if date is None:
                    date = binding['date']['value']
                if 'centity' in binding:
                    claim_entities.add(binding['centity']['value'])
                keywords.add(binding['kw']['value'])

            claim_left_embedding = self.embedder.left_hand_side_entity_embedding(claim).numpy()
            claim_right_embedding = self.embedder.right_hand_side_entity_embedding(claim).numpy()
            to_stack.append(claim_left_embedding)
            to_stack.append(claim_right_embedding)

            mentions_relation_embedding = self.embedder.relation_embedding("http://schema.org/mentions").numpy()
            author_relation_embedding = self.embedder.relation_embedding("http://schema.org/author").numpy()
            item_reviewed_relation_embedding = self.embedder.relation_embedding(
                "http://schema.org/itemReviewed").numpy()

            if author is not None:
                author_embedding = self.embedder.right_hand_side_entity_embedding(author).numpy()
                author_embedding_l = self.embedder.left_hand_side_entity_embedding(author).numpy()
                if self.concat_strategy == NeighbourhoodVectorConcatStrategy.CONCAT_ALL:
                    to_stack.append(author_embedding)
                    to_stack.append(author_relation_embedding)
                    if self.bidirectional:
                        to_stack.append(author_embedding_l)
                else:
                    to_stack.append(((claim_left_embedding * author_relation_embedding) * author_embedding))
                    if self.bidirectional:
                        to_stack.append(((author_embedding_l * author_relation_embedding) * claim_right_embedding))

            if review is not None:
                review_embedding = self.embedder.left_hand_side_entity_embedding(review).numpy()
                review_embedding_r = self.embedder.right_hand_side_entity_embedding(review).numpy()
                if self.concat_strategy == NeighbourhoodVectorConcatStrategy.CONCAT_ALL:
                    to_stack.append(review_embedding)
                    to_stack.append(item_reviewed_relation_embedding)
                    if self.bidirectional:
                        to_stack.append(review_embedding_r)
                else:
                    to_stack.append(((review_embedding * item_reviewed_relation_embedding) * claim_right_embedding))
                    if self.bidirectional:
                        to_stack.append(
                            ((claim_left_embedding * item_reviewed_relation_embedding) * review_embedding_r))

            for claim_mention in claim_entities:
                mention_embedding = self.embedder.right_hand_side_entity_embedding(claim_mention).numpy()
                mention_embedding_l = self.embedder.left_hand_side_entity_embedding(claim_mention).numpy()
                if self.concat_strategy == NeighbourhoodVectorConcatStrategy.CONCAT_ALL:
                    to_stack.append(mention_embedding)
                    to_stack.append(mentions_relation_embedding)
                    if self.bidirectional:
                        to_stack.append(mention_embedding_l)
                else:
                    to_stack.append(((claim_left_embedding * mentions_relation_embedding) * mention_embedding))
                    if self.bidirectional:
                        to_stack.append(((mention_embedding_l * mentions_relation_embedding) * claim_right_embedding))

            to_stack.append(mentions_relation_embedding)

            if review is not None:
                review_embedding = self.embedder.left_hand_side_entity_embedding(review).numpy()
                for review_mention in review_entities:
                    mention_embedding = self.embedder.right_hand_side_entity_embedding(review_mention).numpy()
                    mention_embedding_l = self.embedder.left_hand_side_entity_embedding(review_mention).numpy()
                    if self.concat_strategy == NeighbourhoodVectorConcatStrategy.CONCAT_ALL:
                        to_stack.append(mention_embedding)
                        if self.bidirectional:
                            to_stack.append(mention_embedding_l)

                    else:
                        to_stack.append(((review_embedding * mentions_relation_embedding) * mention_embedding))
                        if self.bidirectional:
                            to_stack.append(
                                ((mention_embedding_l * mentions_relation_embedding) * claim_right_embedding))

            embeddings.append(numpy.hstack(to_stack))

        max_len = 0
        for vector in embeddings:
            vlen = vector.size
            if vlen > max_len:
                max_len = vlen

        padded_embeddings = []
        for vector in embeddings:
            vlen = vector.size
            pad_len = max_len - vlen
            if pad_len > 0:
                padded_embeddings.append(numpy.pad(vector, [(0, pad_len)], mode="constant", constant_values=0))
            else:
                padded_embeddings.append(vector)

        embedding_dataset = numpy.vstack(padded_embeddings)
        return embedding_dataset

    def fit_transform(self, X, y=None, **kwargs):
        """
        perform fit and transform over the data
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        return self.transform(X, y)
