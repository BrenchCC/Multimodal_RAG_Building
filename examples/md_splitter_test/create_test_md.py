import os
import base64
from pathlib import Path


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def create_test_md_files():
    """Create test markdown files with base64 images"""

    # Path to the airplane image
    airplane_path = "examples/airplane.png"

    # Convert image to base64
    base64_image = image_to_base64(airplane_path)

    # Create test directory if it doesn't exist
    test_dir = "examples/md_splitter_test/md_files"
    Path(test_dir).mkdir(parents = True, exist_ok = True)

    # Test file 1: Basic markdown with image
    md_content1 = """# 测试文档 1

这是一个包含图片的测试文档。

## 图片部分

这是飞机的图片：

![飞机图片](data:image/png;base64,{base64_data})

## 文本内容

这是一些示例文本，用于测试文本分割功能。这个文档包含了标题、图片和文本内容，可以用来测试Markdown文件的处理和分割功能。

### 子标题

这里是子标题下的内容，包含更多的文本用于测试分割算法。
""".format(base64_data=base64_image)

    with open(os.path.join(test_dir, "test_doc_01.md"), "w", encoding="utf-8") as f:
        f.write(md_content1)

    # Test file 2: Multiple images and tables
    md_content2 = """# 测试文档 2 - 多图片和表格

这个文档测试多个图片和表格的处理。

## 第一节

这里是第一节的内容。

![飞机图片1](data:image/png;base64,{base64_data})

### 子节内容

这里是子节的内容，后面还有图片。

![飞机图片2](data:image/png;base64,{base64_data})

## 第二节

这里是第二节的内容。

### HTML表格测试

<table>
<thead>
<tr>
<th>名称</th>
<th>描述</th>
<th>值</th>
</tr>
</thead>
<tbody>
<tr>
<td>项目1</td>
<td>描述1</td>
<td>100</td>
</tr>
<tr>
<td>项目2</td>
<td>描述2</td>
<td>200</td>
</tr>
</tbody>
</table>

这里是表格后的文本内容。

![飞机图片3](data:image/png;base64,{base64_data})

## 长文本测试

这里是一些较长的文本内容，用于测试文本分割功能。这个文档包含了多个图片、表格和大量文本，可以全面测试Markdown文件的处理能力。文本分割器需要能够正确处理这些不同类型的内容，并保持文档结构的完整性。

### 更多内容

这里是更多的内容，用于测试分割算法在处理大量文本时的表现。分割器应该能够在保持语义连贯性的同时，将长文本分割成合适大小的块。
""".format(base64_data=base64_image)

    with open(os.path.join(test_dir, "test_doc_02.md"), "w", encoding="utf-8") as f:
        f.write(md_content2)

    # Test file 3: Complex structure
    md_content3 = """# 复杂结构测试文档

这个文档测试复杂的文档结构。

## 主要部分

这是主要部分的内容。

### 子部分A

子部分A的内容，包含图片：

![飞机A](data:image/png;base64,{base64_data})

#### 详细内容A1

详细内容A1，这里有很多文本用于测试分割功能。

#### 详细内容A2

详细内容A2，这里也有大量文本内容。

### 子部分B

子部分B的内容，也包含图片：

![飞机B](data:image/png;base64,{base64_data})

## 另一个主要部分

这是另一个主要部分的内容。

### 表格测试

<table>
<thead>
<tr>
<th>产品</th>
<th>价格</th>
<th>库存</th>
</tr>
</thead>
<tbody>
<tr>
<td>产品A</td>
<td>¥100</td>
<td>50</td>
</tr>
<tr>
<td>产品B</td>
<td>¥200</td>
<td>30</td>
</tr>
<tr>
<td>产品C</td>
<td>¥300</td>
<td>20</td>
</tr>
</tbody>
</table>

### 结论

这里是结论部分，包含最后一张图片：

![飞机C](data:image/png;base64,{base64_data})

这个文档测试了多级标题、多个图片和表格的复杂组合，适合测试分割器处理复杂文档结构的能力。
""".format(base64_data=base64_image)

    with open(os.path.join(test_dir, "test_doc_03.md"), "w", encoding="utf-8") as f:
        f.write(md_content3)

    print(f"测试Markdown文件已创建在: {test_dir}")
    print("创建的文件:")
    for file in ["test_doc_01.md", "test_doc_02.md", "test_doc_03.md"]:
        file_path = os.path.join(test_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  - {file} ({file_size} bytes)")


if __name__ == "__main__":
    create_test_md_files()
