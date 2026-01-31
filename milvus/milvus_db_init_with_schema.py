import os
import sys
import time
import random
import base64
import logging
from typing import List, Optional, Dict

from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pymilvus import DataType, Function, FunctionType, MilvusClient

sys.path.append(os.getcwd())

from utils.common_tools import get_surrounding_text_content
from utils.env_model_util import MILVUS_URI, COLLECTION_NAME
from utils.embeddings_util import LIMITER, RETRY_ON_429, MAX_429_RETRIES, BASE_BACKOFF, convert_item_to_embedding, convert_image_to_base64
from splitter.splitter_tool import MarkdownDirSplitter

logger = logging.getLogger("MilvusVector_Initialization")

class MilvusVector:
    """
    Initialize the Milvus vector database with the given schema.    
    """
    def __init__(self):
        self.vector_stored_saved: Optional[Milvus] = None
        self.client = MilvusClient(uri = MILVUS_URI, user = 'root', password = 'Milvus')

    def create_collection(
        self, 
        collection_name: str = COLLECTION_NAME, 
        uri: str = MILVUS_URI,
        is_first: bool = False
    ):
        """
        Create a collection in the Milvus vector database with the given schema.
        """
        schema = self.client.create_schema()

        # Basic fields  
        schema.add_field("id", DataType.INT64, is_primary = True, auto_id = True, description = "Milvus primary key")
        schema.add_field("category", DataType.VARCHAR, max_length = 1024, description = "Category of the embedding")
        schema.add_field("filename", DataType.VARCHAR, max_length = 1024, description = "Filename of the document")
        schema.add_field("filetype", DataType.VARCHAR, max_length = 1024, description = "Filetype of the document")

        # Document content fields
        schema.add_field("title", DataType.VARCHAR, max_length = 1024, description = "Title of the document")
        schema.add_field("text_content", DataType.VARCHAR, max_length = 65535, description = "Text content of the document")
        schema.add_field("image_path", DataType.VARCHAR, max_length = 1024, description = "Image path of the document")

        # Vector fields 
        schema.add_field("title_sparse", DataType.SPARSE_FLOAT_VECTOR, description = "Parsed title of the document")
        schema.add_field("text_content_sparse", DataType.SPARSE_FLOAT_VECTOR, description = "Sparse vector of text content")
        schema.add_field("text_content_dense", DataType.FLOAT_VECTOR, dim = 1024, description = "Dense vector of text content")

        logger.info(f"Collection {collection_name} created with schema: {schema}, with total {len(schema)} fields")
        
        # Add BM25 function for title field
        title_bm25_function = Function(
            name = "title_bm25",
            input_field_names = ["title"], # the field name of the VARCHAR field to be converted
            output_field_name = "title_sparse", # the field name of the output BM25 vector
            function_type = FunctionType.BM25,
            function_body = "BM25(title)"
        )

        # Add BM25 function for text_content field
        content_bm25_function = Function(
            name = "text_content_bm25_emb",
            input_field_names = ["text_content"], # the field name of the VARCHAR field to be converted
            output_field_name = "text_content_sparse", # the field name of the output BM25 vector
            function_type = FunctionType.BM25,
            function_body = "BM25(text_content)"
        )

        # Add the BM25 function to the schema   
        schema.add_function(title_bm25_function)
        schema.add_function(content_bm25_function)

        try:
            logger.info(f"Start to create index params for collection {collection_name}")
            index_params = self.client.prepare_index_params()

            # Primary index
            index_params.add_index(
                field_name = "id",
                index_type = "AUTOINDEX"
            )

            # Sparse inverted index for title field
            index_params.add_index(
                field_name = "title_sparse",
                index_type = "SPARSE_INVERTED_INDEX",
                metrics_type = "BM25",
                params ={
                    "inverted_index_algo": "DAAT_MAXSCORE", # algorithm for sparse inverted index
                    "bm25_k1": 1.2, # Word Frequency Saturation Control Parameters
                    "bm25_b": 0.75, # Document Length Normalization Parameter
                }
            )

            # Sparse inverted index for text_content field
            index_params.add_index(
                field_name = "text_content_sparse",
                index_type = "SPARSE_INVERTED_INDEX",
                metrics_type = "BM25",
                params ={
                    "inverted_index_algo": "DAAT_MAXSCORE", # algorithm for sparse inverted index
                    "bm25_k1": 1.2, # Word Frequency Saturation Control Parameters
                    "bm25_b": 0.75, # Document Length Normalization Parameter
                }
            )

            # HNSW index for text_content_dense field
            index_params.add_index(
                field_name = "text_content_dense",
                index_type = "HNSW",
                metrics_type = "COSINE",
                params = {
                    "M": 16, # Number of connections for each node in the HNSW graph
                    "efConstruction": 200, # Number of candidates to consider during construction
                }
            )

            logger.info(f"Successfully created index params for collection {collection_name}")

        except Exception as e:
            logger.error(f"Failed to create index params for collection {collection_name}, error: {e}")

        if is_first and COLLECTION_NAME in self.client.list_collections():
            self.client.release_collection(collection_name = COLLECTION_NAME)
            self.client.drop_collection(collection_name = COLLECTION_NAME)

        self.client.create_collection(
            collection_name = COLLECTION_NAME,
            schema = schema,
            index_params = index_params
        )
        logger.info(f"Successfully created collection {collection_name} with schema: {schema}, with total {len(schema)} fields")

    @staticmethod
    def doc_to_dict(docs: List[Document]) -> List[dict]:
        """
        Convert a list of Document objects to a dictionary.

        Args:
            doc (List[Document]): A list of Document objects.

        Returns:
            dict: A dictionary where the keys are the field names and the values are the field values.
        """
        results = []

        for doc in docs:
            doc_dict = {}
            metadata = doc.metadata

            # extract text if embeddings_type == "text"
            embedding_type = metadata.get("embedding_type")
            doc_dict["text"] = doc.page_content if embedding_type == "text" else ""

            # extract category from embedding_type
            doc_dict["category"] = embedding_type

            # extract filename and filetype from source
            source = metadata.get('source', '')
            doc_dict['filename'] = source
            _, ext = os.path.splitext(source)
            doc_dict['filetype'] = ext[1:].lower() if ext.startswith('.') else ext.lower()

            # extract image_path if embeddings_type == "image"
            if embedding_type == "image":
                doc_dict["image_path"] = doc.page_content
                doc_dict["text"] = "image"
            else:
                doc_dict["image_path"] = ""

            # extract the title (concatenating all Headers) and store it in the text field with the content (doc.page_content)
            headers = []
            # Assuming the keys of Header may be 'header_1', 'header_2', 'header_3', etc., we concatenate them in hierarchical order.
            header_keys = [key for key in metadata.keys() if key.startswith('header')]              # ['header_1', 'header_3']
            # Sort the header keys by the number after 'header_' if it exists, otherwise keep the original order
            header_keys_sorted = sorted(header_keys, key = lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else x)

            for key in header_keys_sorted:
                value = metadata.get(key, '').strip()
                if value:  # check if the value is not empty
                    headers.append(value)
            
            # concatenate all non-empty Header values with ' --> '
            doc_dict['title'] = ' --> '.join(headers) if headers else ''  
            # process text_content: concatenate title and content with ':' if title is not empty
            if embedding_type == 'text':
                if doc_dict['title']:
                    doc_dict['text'] = doc_dict['title'] + ':' + doc.page_content
                else:
                    doc_dict['text'] = doc.page_content
            
            # 6. 将doc_dict添加到result_dict中
            results.append(doc_dict)

        return results
    
# TODO: Functions to be implemented: write_to_milvus, generate_image_description, save_to_milvus
