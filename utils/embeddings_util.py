import os
import sys
import time
import base64
import logging
import mimetypes

from http import HTTPStatus
from urllib.parse import urlparse
from typing import List, Tuple, Dict, Optional

import requests
import dashscope

sys.path.append(os.getcwd())
from utils.env_model_util import (
    DASHSCOPE_API_KEY,
    EMBEDDING_MODEL_NAME
)

logger = logging.getLogger("Embeddings_Tool")


# ========= 配置区 =========
DASHSCOPE_MODEL = EMBEDDING_MODEL_NAME  # 指定使用的达摩院多模态嵌入模型名称
DASHSCOPE_API_KEY = DASHSCOPE_API_KEY
RPM_LIMIT = 500  # 每分钟最多调用次数（Requests Per Minute）- 测试用, 已提高限制
WINDOW_SECONDS = 60  # 限流时间窗口（秒）, 与RPM_LIMIT配合实现每分钟限流

RETRY_ON_429 = True  # 是否在遇到429（请求过多）状态码时进行重试
MAX_429_RETRIES = 5  # 429状态码的最大重试次数
BASE_BACKOFF = 2.0  # 指数退避算法的基础等待时间（秒）

# 图片最大体积（URL HEAD 检查）, 若超过则跳过图片项
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3MB
# ======== 配置区结束 =========

# 全局数据容器, 用于存储所有处理后的数据
all_data: List[Dict] = []


class FixedWindowRateLimiter:
    """
    固定窗口速率限制器, 用于控制API调用频率
    """

    def __init__(self, limit: int, window_seconds: int):
        """初始化速率限制器

        Args:
            limit: 时间窗口内允许的最大请求数
            window_seconds: 时间窗口长度（秒）
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.window_start = time.monotonic()  # 当前时间窗口的开始时间
        self.count = 0  # 当前时间窗口内的请求计数

    def acquire(self):
        """获取请求许可, 如果需要会阻塞直到可以继续请求"""
        now_time = time.monotonic()  # 当前时间
        elapsed = now_time - self.window_start

        if elapsed >= self.window_seconds:
            # 如果当前时间超过了时间窗口, 重置计数器和时间窗口开始时间
            self.count = 0
            self.window_start = now_time

        if self.count >= self.limit:
            sleep_sec = self.window_seconds - elapsed
            if sleep_sec > 0:
                logger.info(f"[限速] 达到 {self.limit} 次请求, 等待 {sleep_sec:.2f}s...")
                time.sleep(sleep_sec)  # 阻塞等待
            # 重置计数器
            self.count = 0
            self.window_start = time.monotonic()  # 更新时间窗口开始时间

        self.count += 1


# 初始化固定窗口速率限制器
LIMITER = FixedWindowRateLimiter(RPM_LIMIT, WINDOW_SECONDS)


def is_url(s: str) -> bool:
    try:
        parsed = urlparse(s)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def convert_image_to_base64(img: str) -> Tuple[str, str]:
    """
    支持：
    1. 本地图片路径
    2. 图片 URL

    返回：
    - api_img: data:image/...;base64,... （给 API 用）
    - store_id: 原路径 or 原 URL
    """
    if not img:
        return "", ""

    # URL 图片
    if is_url(img):
        raw = img
        try:
            # HEAD 探测
            try:
                head = requests.head(raw, timeout=5, allow_redirects=True)
            except requests.RequestException as e:
                logger.error(f"[图片] URL HEAD 请求失败：{raw}")
                logger.exception(e)
                return "", ""

            if head.status_code != 200:
                logger.error(f"[图片] URL 不可达, status {head.status_code}：{raw}")
                return "", ""

            size = int(head.headers.get("Content-Length") or 0)
            if size and size > MAX_IMAGE_BYTES:
                logger.warning(f"[图片] URL 大小 {size} > {MAX_IMAGE_BYTES}, 跳过该图：{raw}")
                return "", ""

            mime = head.headers.get("Content-Type")
            if mime:
                mime = mime.split(";")[0]
            else:
                mime = "image/png"

            # GET 真正下载
            resp = requests.get(raw, timeout=10)
            resp.raise_for_status()

            content = resp.content
            if len(content) > MAX_IMAGE_BYTES:
                logger.warning(f"[图片] 实际下载大小超限 {len(content)} > {MAX_IMAGE_BYTES}：{raw}")
                return "", ""

            b64 = base64.b64encode(content).decode("utf-8")
            api_img = f"data:{mime};base64,{b64}"
            return api_img, raw

        except Exception as e:
            logger.error(f"[图片] URL 转 base64 失败：{raw}, 错误：{e}")
            logger.exception(e)
            return "", ""
    # 本地图片
    else:
        try:
            if not os.path.exists(img):
                logger.error(f"[图片] 本地文件不存在：{img}")
                return "", ""

            size = os.path.getsize(img)
            if size > MAX_IMAGE_BYTES:
                logger.warning(f"[图片] 本地文件过大 {size} > {MAX_IMAGE_BYTES}：{img}")
                return "", ""

            mime = mimetypes.guess_type(img)[0] or "image/png"

            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            api_img = f"data:{mime};base64,{b64}"
            return api_img, img

        except Exception as e:
            logger.error(f"[图片] 本地文件转 base64 失败：{img}")
            logger.exception(e)
            return "", ""


def call_dashscope_embedding(input_data: List[Dict]) -> Tuple[bool, List[float], Optional[int], Optional[float]]:
    """调用达摩院多模态嵌入模型

    Args:
        input_data: 输入数据, 每个元素为一个字典, 包含 "image" 和 "text" 键

    Returns:
        Tuple[bool, List[float], Optional[int], Optional[float]]:
            - 第一个元素为是否成功
            - 第二个元素为嵌入向量列表
            - 第三个元素为 HTTP 状态码（如果失败）
            - 第四个元素为延迟时间（如果失败,重试等待）
    """

    # 应用速率限制
    LIMITER.acquire()
    try:
        # 调用达摩院多模态嵌入API
        response = dashscope.MultiModalEmbedding.call(
            model = DASHSCOPE_MODEL,
            input = input_data,
            api_key = DASHSCOPE_API_KEY
        )
    except Exception as e:
        logger.error(f"调用 DashScope 异常：{e}")
        logger.exception(e)
        return False, [], None, None

    # 获取HTTP状态码
    status = getattr(response, "status_code", None)
    retry_after = None

    # 检查是否需要重试等待
    try:
        headers = getattr(response, "headers", None)
        if headers and isinstance(headers, dict):
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra:
                retry_after = float(ra)
    except Exception as e:
        pass

    # 获取API返回的代码和消息
    resp_code = getattr(response, "code", "")
    resp_msg = getattr(response, "message", "")

    # 处理成功响应
    if status == HTTPStatus.OK:
        try:
            # 提取嵌入向量
            embedding = response.output['embeddings'][0]['embedding']
            return True, embedding, status, retry_after
        except Exception as e:
            logger.error(f"解析嵌入失败：{e}")
            logger.exception(e)
            return False, [], status, retry_after
    else:
        # 处理失败响应
        logger.error(f"请求失败, 状态码：{status}, code：{resp_code}, message：{resp_msg}")
        return False, [], status, retry_after


def convert_item_to_embedding(item: Dict) -> Dict:
    """
    将输入项转换为嵌入向量
    mode = 'text'：文本项：把 content 向量化；
    mode = 'image'：图片项：向量化图片

    Args:
        item: 原始数据项
        mode: 处理模式（'text'或'image'）
        api_image: 当mode为'image'时使用的图像数据

    Returns:
        Dict: 处理后的数据项, 包含嵌入向量
    """

    # 创建临时item, 避免修改原始item
    temp_item = item.copy()
    raw_content = (temp_item.get('text') or '').strip()
    image_raw = (temp_item.get('image') or '').strip()

    if image_raw:
        image = convert_image_to_base64(image_raw)[0]
        input_data = [
            {"image": image, "text": raw_content}
        ]
    else:
        input_data = [
            {"text": raw_content}
        ]
        logger.info(f"输入数据：{input_data}")
        logger.info(f"纯文本：{raw_content}")

    success, embedding, status, retry_after = call_dashscope_embedding(
        input_data)

    if success:
        # 这里需要根据你的字段进行处理
        temp_item['text_content_dense'] = embedding
    else:
        temp_item['text_content_dense'] = []

    return temp_item


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 50)
    logger.info("Embeddings Utils 测试")
    logger.info("=" * 50)

    # 测试1: 纯文本嵌入
    logger.info("1. 测试纯文本嵌入:")
    text_item = {
        "text": "这是一个测试文本, 用于验证文本嵌入功能是否正常工作。",
        "image": ""
    }
    result = convert_item_to_embedding(text_item)
    if result['text_content_dense']:
        logger.info(f"✓ 成功获取文本嵌入向量, 维度: {len(result['text_content_dense'])}")
        logger.info(f"✓ 向量前5个值: {result['text_content_dense'][:5]}")
    else:
        logger.error(f"✗ 未能获取文本嵌入向量")

    # 测试2: 图片嵌入（可选测试, 需要提供实际图片路径）
    logger.info("2. 测试图片嵌入:")
    test_image_path = "examples/airplane.png"  # 可以替换为实际测试图片路径
    if os.path.exists(test_image_path):
        image_item = {
            "text": "这是一架飞机",
            "image": test_image_path
        }
        result = convert_item_to_embedding(image_item)
        if result['text_content_dense']:
            logger.info(f"✓ 成功获取图片嵌入向量, 维度: {len(result['text_content_dense'])}")
            logger.info(f"✓ 向量前5个值: {result['text_content_dense'][:5]}")
        else:
            logger.error(f"✗ 未能获取图片嵌入向量")
    else:
        logger.warning(f"⚠ 跳过图片测试: 未找到测试图片 {test_image_path}")

    # 测试3: URL图片嵌入（可选测试, 需要提供可访问的图片URL）
    logger.info("3. 测试URL图片嵌入:")
    test_image_url = "https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch/blob/main/chapter_05_resnet_modern_cnn/images/airplane.png?raw=true"  
    url_item = {
        "text": "这是一张来自URL的测试图片, 是一只猫, 用于测试图片嵌入功能",
        "image": test_image_url
    }
    result = convert_item_to_embedding(url_item)
    if result['text_content_dense']:
        logger.info(f"✓ 成功获取URL图片嵌入向量, 维度: {len(result['text_content_dense'])}")
        logger.info(f"✓ 向量前5个值: {result['text_content_dense'][:5]}")
    else:
        logger.warning(f"⚠ 未能获取URL图片嵌入向量（可能是测试URL不可访问）")

    logger.info("=" * 50)
    logger.info("测试完成")
    logger.info("=" * 50)
