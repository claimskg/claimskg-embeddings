import json
from typing import List

import flair
import numpy
from SPARQLWrapper import SPARQLWrapper, JSON
from flair.data import Sentence, Token
from flair.embeddings import TokenEmbeddings, _get_transformer_sentence_embeddings, DocumentEmbeddings
from kbc.datasets import Dataset
from kbc.embeddings import KnowledgeGraphEmbeddingExtractor
from kbc.models import KBCModel
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from transformers import FlaubertTokenizer, FlaubertModel


class FlaubertEmbeddings(TokenEmbeddings):
    def __init__(
            self,
            pretrained_model_name_or_path: str = "flaubert-large-cased",
            layers: str = "-1",
            pooling_operation: str = "first",
            use_scalar_mix: bool = False,
    ):
        """XLM-RoBERTa as proposed by Conneau et al. 2019.
        :param pretrained_model_name_or_path: name or path of XLM-R model
        :param layers: comma-separated list of layers
        :param pooling_operation: defines pooling operation for subwords
        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
        """
        super().__init__()

        self.tokenizer = FlaubertTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        self.model = FlaubertModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_hidden_states=True,
        )
        self.name = pretrained_model_name_or_path + 'xlm'
        self.layers: List[int] = [int(layer) for layer in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True

        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tokenizer"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # 1-xlm-roberta-large -> xlm-roberta-large
        self.tokenizer = self.tokenizer = FlaubertTokenizer.from_pretrained(
            "-".join(self.name.split("-")[1:])
        )

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        self.model.to(flair.device)
        self.model.eval()

        sentences = _get_transformer_sentence_embeddings(
            sentences=sentences,
            tokenizer=self.tokenizer,
            model=self.model,
            name=self.name,
            layers=self.layers,
            pooling_operation=self.pooling_operation,
            use_scalar_mix=self.use_scalar_mix,
            bos_token="<s>",
            eos_token="</s>",
        )

        return sentences


class FlairTransformer(BaseEstimator, TransformerMixin):
    """
    a general class for creating a machine learning step in the machine learning pipeline
    """

    def __init__(self, embedder: DocumentEmbeddings):
        """
        constructor
        """
        super(FlairTransformer, self).__init__()
        self.embedder = embedder

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

        embeddings = []
        for element in tqdm(X):
            sentence = Sentence(element)
            self.embedder.embed(sentence)

            vector = sentence.get_embedding().cpu().detach().numpy()
            embeddings.append(vector)

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
            # rel_embedding = self.embedder.relation_embedding("http://purl.org/dc/terms/about").numpy()
            # final_embedding = numpy.hstack([claim_embedding, rel_embedding])
            # final_embedding = claim_embedding  # * rel_embedding
            # final_embedding = claim_embedding * rel_embedding
            # embeddings.append(final_embedding)
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


class ClamsKGGraphEmbeddingTransformer(GraphEmbeddingTransformer):
    def __init__(self, dataset: Dataset, model: KBCModel, sparql_endpoint: str):
        super(ClamsKGGraphEmbeddingTransformer, self).__init__(dataset, model)
        self.sparql_endpoint = sparql_endpoint

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
                   schema:datePublished ?date.

              OPTIONAL {{
                  <{element}> schema:mentions ?centity. 
              }}      
            }}
            """

            wrapper.setQuery(sparql_query)
            wrapper.setReturnFormat(JSON)
            result = wrapper.query().response.read()

            strres = str(result, 'utf-8')
            response = json.loads(strres)
            # print(response)

            review_entities = set()
            claim_entities = set()
            review = None
            claim = element
            date = None
            author = None

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
                to_stack.append(((claim_left_embedding * author_relation_embedding) * author_embedding))

            if review is not None:
                review_embedding = self.embedder.left_hand_side_entity_embedding(review).numpy()
                to_stack.append(((review_embedding * item_reviewed_relation_embedding) * claim_right_embedding))

            for claim_mention in claim_entities:
                mention_embedding = self.embedder.right_hand_side_entity_embedding(claim_mention).numpy()
                to_stack.append(((claim_left_embedding * mentions_relation_embedding) * mention_embedding))

            if review is not None:
                for review_mention in review_entities:
                    mention_embedding = self.embedder.right_hand_side_entity_embedding(review_mention).numpy()
                    to_stack.append(((review_embedding * mentions_relation_embedding) * mention_embedding))

            final_embedding = numpy.hstack(to_stack)

            # rel_embedding = self.embedder.relation_embedding("http://purl.org/dc/terms/about").numpy()
            # final_embedding = numpy.hstack([claim_embedding, rel_embedding])
            # final_embedding = claim_embedding  # * rel_embedding
            # final_embedding = claim_embedding * rel_embedding
            # embeddings.append(final_embedding)
            embeddings.append(final_embedding)

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