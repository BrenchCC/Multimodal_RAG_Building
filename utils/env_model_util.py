import os
from dotenv import load_dotenv

# 代理自动检测和处理函数
def configure_proxy_for_local_services():
    """
    自动检测并配置代理设置，确保本地服务（如 Milvus、PaddleOCR）能够正常访问
    当访问 localhost/127.0.0.1 时，应该禁用代理

    Returns:
        dict: 包含代理设置的字典，用于 Milvus 等客户端配置
    """
    import logging
    logger = logging.getLogger("ProxyConfig")

    # 检查是否设置了代理
    has_http_proxy = 'http_proxy' in os.environ or 'HTTP_PROXY' in os.environ
    has_https_proxy = 'https_proxy' in os.environ or 'HTTPS_PROXY' in os.environ

    if has_http_proxy or has_https_proxy:
        logger.debug("代理环境变量已设置，正在配置代理例外...")

        # 对于本地服务，我们需要确保 no_proxy 包含本地地址
        no_proxy = os.environ.get('no_proxy', '')
        no_proxy_upper = os.environ.get('NO_PROXY', '')

        # 合并代理例外列表
        all_no_proxy = set()
        if no_proxy:
            all_no_proxy.update([addr.strip() for addr in no_proxy.split(',') if addr.strip()])
        if no_proxy_upper:
            all_no_proxy.update([addr.strip() for addr in no_proxy_upper.split(',') if addr.strip()])

        # 确保本地地址在代理例外中
        local_addresses = ['localhost', '127.0.0.1', '0.0.0.0']
        for addr in local_addresses:
            if addr not in all_no_proxy:
                all_no_proxy.add(addr)
                logger.debug(f"添加本地地址到代理例外: {addr}")

        # 更新环境变量
        updated_no_proxy = ','.join(all_no_proxy)
        os.environ['no_proxy'] = updated_no_proxy
        os.environ['NO_PROXY'] = updated_no_proxy
        logger.debug(f"更新后的代理例外列表: {updated_no_proxy}")

        return {
            'http_proxy': None,
            'https_proxy': None,
            'no_proxy': updated_no_proxy
        }
    else:
        logger.debug("未检测到代理环境变量")
        return {}

# 加载 .env 文件
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import VolcanoEmbeddings


# 直接从环境变量读取配置
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VOLCES_BASE_URL = os.getenv("VOLCES_BASE_URL", "")
VOLCES_API_KEY = os.getenv("VOLCES_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "")
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "")
SILICON_BASE_URL = os.getenv("SILICON_BASE_URL", "")
SILICON_API_KEY = os.getenv("SILICON_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

# Milvus 配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multimodal_rag")

# PaddleOCR 配置
PADDLE_OCR_URI = os.getenv("PADDLE_OCR_URI", "localhost:8118")

# LLM 配置
LLM_NAME = os.getenv("LLM_NAME", "gpt-4o-mini")
LLM_IDENTIFIER = os.getenv("LLM_IDENTIFIER", "GPT-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "multimodal-embedding-v1")


class LLMClient:
    """LLM 客户端工厂类，用于初始化和管理不同类型的 LLM 客户端"""

    @staticmethod
    def get_openai_llm(
        model: str = "gpt-4o-mini",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取 OpenAI 风格的 LLM 客户端"""
        api_key = api_key or OPENAI_API_KEY
        base_url = base_url or OPENAI_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )

    @staticmethod
    def get_openai_embedding(
        model: str = "text-embedding-3-small",
        api_key: str = None,
        dimensions: int = 1024,
        **kwargs
    ) -> OpenAIEmbeddings:
        """获取 OpenAI 风格的 Embedding 模型"""
        api_key = api_key or OPENAI_API_KEY

        return OpenAIEmbeddings(
            model = model,
            openai_api_key = api_key,
            dimensions = dimensions,
            **kwargs
        )

    @staticmethod
    def get_volces_llm(
        model: str = "ep-20251205135937-7dhcx",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取 Volces 模型客户端"""
        api_key = api_key or VOLCES_API_KEY
        base_url = base_url or VOLCES_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )

    @staticmethod
    def get_qwen_embedding(
        model: str = "text-embedding-v4",
        api_key: str = None,
        **kwargs
    ) -> DashScopeEmbeddings:
        """获取 DashScope（Qwen）Embedding 模型"""
        api_key = api_key or DASHSCOPE_API_KEY

        return DashScopeEmbeddings(
            model = model,
            dashscope_api_key = api_key,
            **kwargs
        )

    @staticmethod
    def get_qwen_llm(
        model: str = "qwen-omni-turbo",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取 Qwen 模型客户端"""
        api_key = api_key or DASHSCOPE_API_KEY
        base_url = base_url or DASHSCOPE_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )

    @staticmethod
    def get_qwen3_llm(
        model: str = "qwen3-235b-a22b",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = True,
        enable_search: bool = True,
        **kwargs
    ) -> ChatOpenAI:
        """获取 Qwen3 模型客户端（支持搜索功能）"""
        api_key = api_key or DASHSCOPE_API_KEY
        base_url = base_url or DASHSCOPE_BASE_URL

        extra_body = {}
        if enable_search:
            extra_body = {
                "enable_search": True,
                "search_options": {
                    "forced_search": False
                },
                "enable_thinking": False
            }

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            extra_body = extra_body,
            **kwargs
        )

    @staticmethod
    def get_deepseek_llm(
        model: str = "deepseek-chat",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取 DeepSeek 模型客户端"""
        api_key = api_key or DEEPSEEK_API_KEY
        base_url = base_url or DEEPSEEK_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )

    @staticmethod
    def get_glm_llm(
        model: str = "glm-4-air-250414",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取 GLM 模型客户端"""
        api_key = api_key or GLM_API_KEY
        base_url = base_url or GLM_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )

    @staticmethod
    def get_glm_multimodal_llm(
        model: str = "glm-4v-plus-0111",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取 GLM 多模态模型客户端"""
        api_key = api_key or GLM_API_KEY
        base_url = base_url or GLM_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )

    @staticmethod
    def get_moonshot_llm(
        model: str = "moonshot-v1-8k",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取 MoonShot 模型客户端"""
        api_key = api_key or MOONSHOT_API_KEY
        base_url = base_url or MOONSHOT_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )

    @staticmethod
    def get_silicon_llm(
        model: str = "Qwen/Qwen2-7B-Instruct",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取 SiliconFlow 模型客户端"""
        api_key = api_key or SILICON_API_KEY
        base_url = base_url or SILICON_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )

    @staticmethod
    def get_local_llm(
        model: str = "qwen-omni-7b",
        api_key: str = None,
        base_url: str = None,
        streaming: bool = False,
        **kwargs
    ) -> ChatOpenAI:
        """获取本地部署的模型客户端"""
        api_key = api_key or LOCAL_API_KEY
        base_url = base_url or LOCAL_BASE_URL

        return ChatOpenAI(
            model = model,
            api_key = api_key,
            base_url = base_url,
            streaming = streaming,
            **kwargs
        )


# 全局 LLM 客户端实例（可根据需要修改默认配置）
llm_client = LLMClient()

# 常用模型的快捷访问
def get_default_llm() -> ChatOpenAI:
    """获取默认的 LLM 客户端"""
    return llm_client.get_openai_llm()

def get_default_embedding() -> OpenAIEmbeddings:
    """获取默认的 Embedding 模型"""
    return llm_client.get_openai_embedding()

def get_multimodal_llm() -> ChatOpenAI:
    """获取多模态模型客户端"""
    return llm_client.get_qwen_llm(model = "qwen-omni-turbo")


if __name__ == "__main__":
    # 测试代码
    print("LLM Client 初始化测试")
    print("-" * 50)

    # 测试 OpenAI LLM
    try:
        openai_llm = llm_client.get_openai_llm()
        print("✓ OpenAI LLM 初始化成功")
    except Exception as e:
        print(f"✗ OpenAI LLM 初始化失败: {e}")

    # 测试 OpenAI Embedding
    try:
        openai_emb = llm_client.get_openai_embedding()
        print("✓ OpenAI Embedding 初始化成功")
    except Exception as e:
        print(f"✗ OpenAI Embedding 初始化失败: {e}")

    # 测试 Qwen LLM
    try:
        qwen_llm = llm_client.get_qwen_llm()
        print("✓ Qwen LLM 初始化成功")
    except Exception as e:
        print(f"✗ Qwen LLM 初始化失败: {e}")

    # 测试 Qwen Embedding
    try:
        qwen_emb = llm_client.get_qwen_embedding()
        print("✓ Qwen Embedding 初始化成功")
    except Exception as e:
        print(f"✗ Qwen Embedding 初始化失败: {e}")

    print("-" * 50)
    print("测试完成")
