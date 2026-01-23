import os
import sys

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import VolcanoEmbeddings

sys.path.append(os.getcwd())
from utils.model_config_utils import MODEL_CONFIGS



OPENAI_BASE_URL, OPENAI_API_KEY = MODEL_CONFIGS.OPENAI_BASE_URL, MODEL_CONFIGS.OPENAI_API_KEY
VOLCES_BASE_URL, VOLCES_API_KEY = MODEL_CONFIGS.VOLCES_BASE_URL, MODEL_CONFIGS.VOLCES_API_KEY
DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY = MODEL_CONFIGS.DASHSCOPE_BASE_URL, MODEL_CONFIGS.DASHSCOPE_API_KEY
DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY = MODEL_CONFIGS.DEEPSEEK_BASE_URL, MODEL_CONFIGS.DEEPSEEK_API_KEY
GLM_BASE_URL, GLM_API_KEY = MODEL_CONFIGS.GLM_BASE_URL, MODEL_CONFIGS.GLM_API_KEY
MOONSHOT_BASE_URL, MOONSHOT_API_KEY = MODEL_CONFIGS.MOONSHOT_BASE_URL, MODEL_CONFIGS.MOONSHOT_API_KEY
LOCAL_BASE_URL, LOCAL_API_KEY = MODEL_CONFIGS.LOCAL_BASE_URL, MODEL_CONFIGS.LOCAL_API_KEY

# OpenAI Model Initialize
llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    api_key = OPENAI_API_KEY,
    base_url = OPENAI_BASE_URL,
)
# OpenAI Embedding Model Initialize
openai_embedding = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    openai_api_key = OPENAI_API_KEY,
    dimensions = 1024  # set dimension to 1024 for Milvus collection schema
    )

# 
volces_llm = ChatOpenAI(
    model = 'ep-20251205135937-7dhcx', # Doubao Seed 1.6
    api_key = VOLCES_API_KEY,
    base_url = VOLCES_BASE_URL,
)

# DashScope Embedding Model Initialize(Qwen)
qwen_embeddings = DashScopeEmbeddings(
    model = "text-embedding-v4", 
    dashscope_api_key = DASHSCOPE_API_KEY,
)
 
# Qwen Multimodal Model Initialize
multiModal_llm = ChatOpenAI(
    model = 'qwen-omni-turbo',
    api_key = DASHSCOPE_API_KEY,
    base_url = DASHSCOPE_BASE_URL,
    streaming = True
)

# Local Multimodal Model Initialize
# multiModal_llm = ChatOpenAI(
#     model = 'qwen-omni-7b',
#     api_key = LOCAL_API_KEY,
#     base_url = LOCAL_BASE_URL,
# )

# Qwen Series Model Initialize
qwen3 = ChatOpenAI(
    model = 'qwen3-235b-a22b',
    api_key = DASHSCOPE_API_KEY,
    base_url = DASHSCOPE_BASE_URL,
    streaming = True,
    extra_body={
        "enable_search": True,  # open search
        "search_options": {
            "forced_search": False
        },
        "enable_thinking": False  # close thinking
    }
)

qwen3_max = ChatOpenAI(
    model = "qwen3-max",
    api_key = DASHSCOPE_API_KEY,
    base_url = DASHSCOPE_BASE_URL,
    streaming = True,
    extra_body={
        "enable_search": True,  # open search
        "search_options": {
            "forced_search": False
        },
        "enable_thinking": False  # close thinking
    }
)

llm = ChatOpenAI(
    model='qwen3-32b',
    api_key = DASHSCOPE_API_KEY,
    base_url = DASHSCOPE_BASE_URL,
    streaming = True,
    extra_body={
        "enable_search": True,  # open search
        "search_options": {
            "forced_search": False
        },
        "enable_thinking": False  # close thinking
    }
)

# deepseek
llm = ChatOpenAI(
    model = 'deepseek-chat',
    # model='deepseek-reasoner'
    api_key = DEEPSEEK_API_KEY,
    base_url = DEEPSEEK_BASE_URL,
)

# glm-asr 模型
llm = ChatOpenAI(
    # model='glm-asr',
    model = 'glm-4-air-250414',
    api_key = GLM_API_KEY,
    base_url = GLM_BASE_URL,
)


# # GLM 全模态
multiModal_llm = ChatOpenAI(
    model = 'glm-4v-plus-0111',
    api_key = GLM_API_KEY,
    base_url = GLM_BASE_URL,
)