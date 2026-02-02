import os
import sys
import time
import random
import base64
import logging
import argparse
from typing import List, Optional, Dict

from tqdm import tqdm
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pymilvus import DataType, Function, FunctionType, MilvusClient

sys.path.append(os.getcwd())

from utils.common_tools import get_surrounding_text_content
from utils.env_model_util import LLMClient
from utils.env_model_util import MILVUS_URI, COLLECTION_NAME
from utils.env_model_util import configure_proxy_for_local_services
from utils.embeddings_util import LIMITER, RETRY_ON_429, MAX_429_RETRIES, BASE_BACKOFF, convert_item_to_embedding, convert_image_to_base64
from splitter.splitter_tool import MarkdownDirSplitter

logger = logging.getLogger("MilvusVectorDB_Initialization")

qwen_llm = LLMClient.get_qwen_llm(
    model = "qwen-max-latest"
)

class MilvusVectorDB:
    """
    Initialize the Milvus vector database with the given schema.    
    """
    def __init__(self):
        self.vector_stored_saved: Optional[Milvus] = None
        # Milvus 连接配置 - 自动处理代理设置，确保本地服务可访问
        logger.info("Configuring proxy settings for Milvus connection...")
        proxy_config = configure_proxy_for_local_services()

        # Milvus 连接配置 - 注意：默认情况下 Milvus 可能没有启用认证，或者密码不同
        try:
            logger.info(f"Attempting to connect to Milvus at: {MILVUS_URI}")

            # 尝试无密码连接（Milvus 默认可能没有启用认证）
            # 对于本地 Milvus 服务，我们需要确保不使用代理
            self.client = MilvusClient(uri = MILVUS_URI)
            logger.info("Successfully connected to Milvus (no authentication)")
        except Exception as e:
            logger.warning(f"Connection without authentication failed: {e}")
            logger.info("Attempting to connect with default credentials (root:Milvus)")
            # 尝试使用默认密码连接
            self.client = MilvusClient(uri = MILVUS_URI, user = 'root', password = 'Milvus')
            logger.info("Successfully connected to Milvus with default credentials")

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
        schema.add_field("title", DataType.VARCHAR, max_length = 1024, description = "Title of the document", enable_analyzer = True)
        schema.add_field("text", DataType.VARCHAR, max_length = 65535, description = "Text content of the document", enable_analyzer = True)
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
            output_field_names = ["title_sparse"], # the field name of the output BM25 vector (plural)
            function_type = FunctionType.BM25
        )

        # Add BM25 function for text_content field
        content_bm25_function = Function(
            name = "text_content_bm25_emb",
            input_field_names = ["text"], # the field name of the VARCHAR field to be converted
            output_field_names = ["text_content_sparse"], # the field name of the output BM25 vector (plural)
            function_type = FunctionType.BM25
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
                metric_type = "BM25",  
                inverted_index_algo = "DAAT_MAXSCORE",
                bm25_k1 = 1.2,
                bm25_b = 0.75
            )

            # Sparse inverted index for text_content field 
            index_params.add_index(
                field_name = "text_content_sparse",
                index_type = "SPARSE_INVERTED_INDEX",
                metric_type = "BM25",  
                inverted_index_algo = "DAAT_MAXSCORE",
                bm25_k1 = 1.2,
                bm25_b = 0.75
            )

            # HNSW index for text_content_dense field - 使用正确的参数格式
            index_params.add_index(
                field_name = "text_content_dense",
                index_type = "HNSW",
                metric_type = "COSINE",  # 使用单数 metric_type 而非 metrics_type
                M = 16,
                efConstruction = 200
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

    def write_to_milvus(self, processed_data: List[dict]):
        """
        Write processed data to Milvus collection.

        Args:
            processed_data (List[dict]): A list of dictionaries, each containing processed data for a document.
        """
        if not processed_data:
            logger.warning("No processed data to write to Milvus.")
            return

        # Truncate text to MAX_TEXT_LENGTH
        MAX_TEXT_LENGTH = 10000
        for item in processed_data:
            text = item.get("text", "")
            if len(text) > MAX_TEXT_LENGTH:
                item["text"] = text[:MAX_TEXT_LENGTH]
                logger.warning(f"Text truncated to {MAX_TEXT_LENGTH} characters for item {item.get('filename', 'unknown')}")

        try:
            insert_data = self.client.insert(collection_name = COLLECTION_NAME, data = processed_data)
            logger.info(f"Successfully inserted {len(insert_data)} items into collection {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to insert data into collection {COLLECTION_NAME}, error: {e}")
            raise e
    
    @staticmethod
    def _read_prompt(prompt_name: str) -> str:
        """
        Read prompt from prompts directory.

        Args:
            prompt_name (str): Name of the prompt file (without .md extension)

        Returns:
            str: Prompt content
        """
        prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../prompts",
            f"{prompt_name}.md"
        )
        try:
            with open(prompt_path, "r", encoding = "utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read prompt {prompt_name}: {e}")
            # Return a default prompt if file not found
            return """You are an expert in scientific paper image understanding. Please analyze this image and generate an English description.
Focus on identifying the image type, core content, and key information. If it's a chart, describe axes and trends. If it's an architecture/flow diagram, explain the main components. Keep the description between 200-400 words."""

    @staticmethod
    def generate_image_description(data_list: List[Dict]):
        """
        Generate image description for each item in data_list.

        Args:
            data_list (List[Dict]): A list of dictionaries, each containing image data.

        Returns:
            List[Dict]: A list of dictionaries, each containing image description.
        """

        # Count images to process for better progress tracking
        image_count = sum(1 for item in data_list if item.get("image_path"))
        if image_count > 0:
            logger.info(f"Found {image_count} images to process")

            for index, item in enumerate(tqdm(data_list, desc = "Generating image descriptions", unit = "image"), 1):
                image_path = item.get("image_path", "")
                if image_path:
                    prev_text, next_text = get_surrounding_text_content(data_list, index)

                    logger.info("=" * 50)
                    logger.info(f"Processing image item {index}: {image_path}")
                    logger.info(f"prev_text: {prev_text[:100] if prev_text else 'No previous text'}")
                    logger.info(f"next_text: {next_text[:100] if next_text else 'No next text'}")

                    base64_image, _ = convert_image_to_base64(image_path)

                    # Select appropriate prompt based on context
                    if prev_text and next_text:
                        system_prompt = MilvusVectorDB._read_prompt("gen_Image_description_full").format(
                            prev_text = prev_text,
                            next_text = next_text
                        )
                    elif prev_text:
                        system_prompt = MilvusVectorDB._read_prompt("gen_image_decription_prev").format(
                            prev_text = prev_text
                        )
                    elif next_text:
                        system_prompt = MilvusVectorDB._read_prompt("gen_image_description_next").format(
                            next_text = next_text
                        )
                    else:
                        system_prompt = MilvusVectorDB._read_prompt("gen_Image_description_null")

                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{base64_image}"
                                }
                            }
                        ]
                    )

                    response = qwen_llm.invoke([message])
                    item["text"] = response.content

                    logger.info(f"Generated description: {item['text'][:100]}...")
        else:
            logger.info("No images to process")

        return data_list
    
    def save_to_milvus(self, processed_data: List[Document]):
        """
        Save processed data to Milvus vector database.

        Args:
            processed_data (List[Document]): A list of processed documents.
        """

        expanded_data = MilvusVectorDB.generate_image_description(MilvusVectorDB.doc_to_dict(processed_data))
        processed_data: List[Dict] = []

        logger.info(f"Converting {len(expanded_data)} items to embeddings...")

        for idx, item in enumerate(tqdm(expanded_data, desc = "Converting to embeddings", unit = "item"), 1):
            # limit the number of requests per second
            LIMITER.acquire()

            if RETRY_ON_429:
                attempts = 0
                while True:
                    result = convert_item_to_embedding(item.copy())

                    if result.get("text_content_dense"):
                        processed_data.append(result)
                        break
                    attempts += 1
                    if attempts > MAX_429_RETRIES:
                        logger.error(f"Exceeded {MAX_429_RETRIES} attempts for item {idx}, skipping this item")
                        processed_data.append(result)
                        break
                    backoff = BASE_BACKOFF * (2 ** (attempts - 1)) * (0.8 + random.random() * 0.4)
                    logger.info(f"Retry {attempts} times for item {idx}, waiting {backoff:.2f} seconds")
                    time.sleep(backoff)
            else:
                result = convert_item_to_embedding(item.copy())
                processed_data.append(result)

        self.write_to_milvus(processed_data)

        return processed_data
    
def initialize_milvus(data_list: List[Document]) -> List[Dict]:
    """
    Initialize Milvus vector database with data_list.

    Args:
        data_list (List[Document]): A list of documents to be initialized.

    Returns:
        List[Dict]: A list of dictionaries, each containing processed data.
    """

    milvus_vector_db = MilvusVectorDB()
    processed_data = milvus_vector_db.save_to_milvus(data_list)
    return processed_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description = "Milvus Vector DB Initialization and Testing")
    parser.add_argument("--test_dir", type = str, default = "examples/md_splitter_test/md_files/", help = "Directory of MD files")
    parser.add_argument("--output_dir", type = str, default = "examples/md_splitter_test/output/images", help = "Image output directory")
    parser.add_argument("--source_name", type = str, default = "LLM.md", help = "Source filename for metadata")
    parser.add_argument("--recreate_collection", action = "store_true", help = "Recreate collection if it exists")
    parser.add_argument("--log_level", type = str, default = "INFO", choices = ["DEBUG", "INFO", "WARNING", "ERROR"], help = "Logging level")

    return parser.parse_args()

if __name__ == "__main__":
    # Configure logging
    args = parse_args()
    logging.basicConfig(
        level = getattr(logging, args.log_level.upper()),
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    logger.info("=" * 80)
    logger.info("Milvus Vector DB Initialization Test")
    logger.info("=" * 80)

    try:
        # Initialize Milvus Vector DB
        milvus_vector_db = MilvusVectorDB()

        # Create collection (with option to recreate)
        logger.info("-" * 60)
        logger.info("Creating Milvus collection")
        logger.info("-" * 60)
        milvus_vector_db.create_collection(is_first = args.recreate_collection)

        # Process markdown files
        logger.info("-" * 60)
        logger.info(f"Processing markdown files from: {args.test_dir}")
        logger.info("-" * 60)

        splitter = MarkdownDirSplitter(images_output_dir = args.output_dir)
        documents = splitter.process_md_dir(args.test_dir, source_filename = args.source_name)[:5]

        logger.info(f"Successfully processed {len(documents)} documents")

        # Save to Milvus
        logger.info("-" * 60)
        logger.info("Saving data to Milvus")
        logger.info("-" * 60)

        processed_data = milvus_vector_db.save_to_milvus(documents)

        # Display results
        logger.info("-" * 60)
        logger.info("Processing completed")
        logger.info("-" * 60)

        logger.info(f"Total items processed: {len(processed_data)}")

        # Count different types of items
        text_count = sum(1 for item in processed_data if item.get("category") == "text")
        image_count = sum(1 for item in processed_data if item.get("category") == "image")

        logger.info(f"Text items: {text_count}")
        logger.info(f"Image items: {image_count}")

        # Display sample items
        if processed_data:
            logger.info("-" * 60)
            logger.info("Sample processed items (first 3):")
            logger.info("-" * 60)

            for i, item in enumerate(processed_data[:3]):
                logger.info(f"\nItem {i + 1}:")
                logger.info(f"  Category: {item.get('category')}")
                logger.info(f"  Filename: {item.get('filename')}")
                logger.info(f"  Filetype: {item.get('filetype')}")
                if item.get("title"):
                    logger.info(f"  Title: {item.get('title')}")
                if item.get("text"):
                    logger.info(f"  Text (preview): {item.get('text')[:200]}...")
                if item.get("image_path"):
                    logger.info(f"  Image path: {item.get('image_path')}")

        logger.info("=" * 80)
        logger.info("Test completed successfully")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Test failed: {str(e)}")
        logger.error("=" * 80)
        logger.exception("Exception details:")
        sys.exit(1)
        

