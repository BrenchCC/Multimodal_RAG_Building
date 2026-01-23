# MarkdownDirSplitter 架构文档

## 概述

MarkdownDirSplitter 是一个多功能的 Markdown 文档处理工具，专为处理包含多种内容类型（文本、图像、HTML 表格）的复杂文档而设计。它提供了完整的文档解析、清洗、分块和语义处理功能，适用于构建多模态 RAG（检索增强生成）系统。

该工具支持批量处理目录中的 Markdown 文件，提取并保存 Base64 编码的图像，将 HTML 表格转换为 Markdown 格式，并使用分层标题和语义切分技术将文档分割成适合嵌入和检索的 chunks。

## 架构设计

```
MarkdownDirSplitter
├── 初始化配置
│   ├── images_output_dir: 图像输出目录
│   ├── text_chunk_size: 文本分块阈值
│   ├── headers_to_split_on: 标题分割规则
│   ├── text_splitter: Markdown标题切割器
│   └── semantic_splitter: 语义切割器
│
├── 图像处理模块
│   ├── save_base64_to_image(): Base64图像解码保存
│   ├── extract_and_replace_images(): 处理Markdown中的Base64图像
│   └── 图像Document创建与元数据管理
│
├── 文档处理模块
│   ├── convert_html_table_to_markdown(): HTML表格转Markdown
│   ├── convert_html_to_markdown(): 批量HTML表格处理
│   ├── process_md_file(): 单个Markdown文件处理
│   ├── process_md_dir(): 目录批量处理
│   └── add_title_hierarchy(): 标题层级补充
│
└── 输出模块
    └── Document列表 (包含文本和图像Document)
```

## 核心功能

### 1. 图像处理
- **Base64 图像提取与保存**：从 Markdown 中提取内嵌的 Base64 编码图像，解码后保存到指定目录
- **图像哈希命名**：使用 MD5 哈希生成唯一文件名，避免重复图像存储
- **图像文档创建**：为每个图像创建单独的 Document 对象，包含源文件、替代文本和嵌入类型信息
- **图像占位符**：将提取后的图像替换为 `[image]` 占位符，保持文本结构完整性

### 2. 文档解析与转换
- **HTML 表格转换**：将 HTML 格式的表格转换为 Markdown 表格
- **容错处理**：对格式不正确的 HTML 表格提供容错支持
- **表格结构优化**：处理表头提取、数据行处理和列宽调整

### 3. 分层分块策略
- **标题切分**：基于 Markdown 标题（#、##、###）进行结构化切分
- **语义切分**：使用递归字符切分器处理过长文本块
- **分块大小控制**：支持自定义文本分块大小和重叠度
- **标题层级补充**：处理跨文件边界的标题层级继承问题

### 4. 批量处理
- **目录遍历**：递归处理目录中的所有 Markdown 文件
- **文件排序**：确保按照逻辑顺序处理文件（如 page_01.md、page_02.md）
- **元数据管理**：为每个文档块添加完整的元数据信息（源文件、标题层次、嵌入类型）

## 数据结构

### Document 对象

`MarkdownDirSplitter` 使用 LangChain 的 `Document` 对象作为核心数据结构：

```python
from langchain_core.documents import Document

# 文本Document
text_doc = Document(
    page_content="文本内容",
    metadata={
        "source": "源文件名",
        "header_1": "一级标题",
        "header_2": "二级标题",
        "header_3": "三级标题",
        "embedding_type": "text"
    }
)

# 图像Document
image_doc = Document(
    page_content="/path/to/image.png",
    metadata={
        "source": "源文件名",
        "alt_text": "图像描述",
        "embedding_type": "image"
    }
)
```

## 关键算法

### 1. 图像提取算法
```python
# 正则表达式匹配 Markdown 图像语法
pattern = r'!\[(.*?)\]\(data:image/(.*?);base64,(.*?)\)'
```

### 2. 表格转换算法
- 使用 BeautifulSoup 解析 HTML 表格
- 智能识别表头（<thead> 或首行 <th>）
- 处理表格数据行并转换为 Markdown 格式
- 处理列数不匹配的容错情况

### 3. 分块算法
```python
# 1. 标题切分（使用 MarkdownHeaderTextSplitter）
# 2. 语义切分（使用 RecursiveCharacterTextSplitter）
# 3. 分块大小控制：递归切分过长文本块
```

### 4. 标题层级继承算法
```python
# 使用状态追踪器维护当前标题层次
current_titles = {1: "", 2: "", 3: ""}
# 为缺失标题的文档继承当前层次的标题信息
```

## 使用场景

### 主要应用

1. **多模态 RAG 系统**
   - 处理包含图表、公式的技术文档
   - 构建支持图像检索的知识库
   - 处理科学论文和报告

2. **文档索引与检索**
   - 将大型文档分割成可管理的 chunk
   - 为每个 chunk 添加语义元数据
   - 支持基于内容的文档检索

3. **内容迁移与转换**
   - 将 HTML 内容转换为 Markdown
   - 清洗和规范化文档格式
   - 批量处理大量文档

## 配置参数

### 初始化参数

| 参数名称 | 类型 | 默认值 | 描述 |
|---------|------|--------|------|
| `images_output_dir` | `str` | 必填 | 提取图像的保存目录 |
| `text_chunk_size` | `int` | 1000 | 文本分块大小阈值（字符数） |

### 内部配置

```python
# 标题切分规则
headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3")
]

# 语义切分参数
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)
```

## 可扩展性

### 1. 文本切分器扩展

可以通过替换以下组件实现自定义切分策略：

```python
class CustomMarkdownDirSplitter(MarkdownDirSplitter):
    def __init__(self, images_output_dir, text_chunk_size):
        super().__init__(images_output_dir, text_chunk_size)
        # 替换为自定义切分器
        self.header_splitter = CustomHeaderSplitter()
        self.recursive_splitter = CustomSemanticSplitter()
```

### 2. 图像处理扩展

```python
class CustomMarkdownDirSplitter(MarkdownDirSplitter):
    def save_base64_to_image(self, base64_str, output_path):
        # 自定义图像解码与保存逻辑
        pass

    def extract_and_replace_images(self, markdown_content, source):
        # 自定义图像提取逻辑
        pass
```

### 3. 表格转换扩展

```python
class CustomMarkdownDirSplitter(MarkdownDirSplitter):
    def convert_html_table_to_markdown(self, html_table):
        # 自定义 HTML 表格转换逻辑
        pass
```

## 注意事项

### 1. 性能优化

- **图像处理**：大量 Base64 图像会显著增加处理时间
- **分块策略**：过小的 `text_chunk_size` 会导致过多的文档块
- **内存管理**：处理大型文档时建议分批次处理

### 2. 格式兼容性

- **HTML 表格**：复杂嵌套表格可能无法完美转换
- **Base64 图像**：仅支持标准图像格式（PNG、JPEG、JPG）
- **标题层次**：非标准标题格式（如 === 下划线）不被支持

### 3. 错误处理

- 缺失文件会记录警告日志并继续处理其他文件
- 图像解码失败会抛出异常，建议使用 try-catch 捕获
- HTML 表格转换失败会保留原始 HTML 内容

## 使用方法

### 基础使用

```python
from splitter.splitter_tool import MarkdownDirSplitter

# 初始化切分器
splitter = MarkdownDirSplitter(
    images_output_dir="/path/to/images",
    text_chunk_size=1000
)

# 处理单个文件
documents = splitter.process_md_file("document.md")

# 处理目录
documents = splitter.process_md_dir(
    md_dir="/path/to/markdown/files",
    source_filename="document_collection"
)

# 查看结果
for doc in documents:
    print(f"Type: {doc.metadata['embedding_type']}")
    print(f"Content: {doc.page_content[:50]}...")
    print(f"Headers: {doc.metadata.get('header_1', 'N/A')}")
    print("-" * 50)
```

### 自定义模块导入

确保项目根目录在 Python 路径中：

```python
import sys
import os
sys.path.append(os.getcwd())

# 然后导入
from splitter.splitter_tool import MarkdownDirSplitter
```

### 命令行运行

```bash
python splitter_tool.py \
    --test_dir examples/md_splitter_test/md_files/ \
    --output_dir examples/md_splitter_test/output/images \
    --source_name LLM.md
```

## 工作流程

```
输入: Markdown 目录
├─1. 遍历目录，按顺序加载所有 .md 文件
├─2. 对每个文件执行以下操作：
│  ├─a. 读取文件内容
│  ├─b. 转换 HTML 表格为 Markdown
│  ├─c. 按标题层级切分文档
│  ├─d. 提取并保存 Base64 图像
│  ├─e. 为切分后的块添加元数据
│  └─f. 对过长文本块进行语义切分
├─3. 跨文件补充标题层级信息
└─输出: 包含文本和图像 Document 的列表
```

## 依赖与安装

### 核心依赖

```
langchain-core
langchain-text-splitters
Pillow
beautifulsoup4
```

### 安装命令

```bash
pip install langchain-core langchain-text-splitters pillow beautifulsoup4
```

## 测试与验证

项目包含测试示例：

```
examples/md_splitter_test/
├─ md_files/          # 测试用 Markdown 文件
└─ output/            # 输出目录
   └─ images/         # 提取的图像文件
```

运行测试：

```bash
python splitter_tool.py
```

输出前 5 个文档的预览信息。
