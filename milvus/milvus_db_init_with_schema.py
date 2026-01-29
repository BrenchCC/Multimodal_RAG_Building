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
