import os
import io
import re
import sys
import base64
import logging
import hashlib
import argparse
from typing import List, Tuple, Optional, Any

from PIL import Image
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

sys.path.append(os.getcwd())

from utils.common_tools import get_sorted_md_files

logger = logging.getLogger("splitter_tool")

class MarkdownDirSplitter:
    """
    A tool module for processing, cleaning, and splitting Markdown files from a directory.
    It handles base64 image extraction, HTML table conversion, and hierarchical text splitting.
    """

    def __init__(
        self,
        images_output_dir: str,
        text_chunk_size: int = 1000
    ):
        """
        Initialize the MarkdownDirSplitter.

        :param images_output_dir: Directory to save extracted base64 images.
        :param text_chunk_size: Threshold for text splitting (characters).
        """
        self.images_output_dir = images_output_dir
        self.text_chunk_size = text_chunk_size
        os.makedirs(self.images_output_dir, exist_ok = True)

        # Define header levels for splitting
        self.headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]

        # Initialize MarkdownHeaderTextSplitter
        self.header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = self.headers_to_split_on)

        # Initialize Semantic/Recursive Splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.text_chunk_size,
            chunk_overlap = 200,
            length_function = len,
            is_separator_regex = False
        )

    @staticmethod
    def save_base64_to_image(
        base64_str: str,
        output_path: str
    ) -> None:
        """
        Decode a base64 string and save it as an image file.

        :param base64_str: The base64 encoded string (pure data).
        :param output_path: The file path to save the image.
        """
        try:
            # Handle data URL prefix if present (though regex usually splits it, this is a safety net)
            if "base64," in base64_str:
                base64_str = base64_str.split("base64,")[1]

            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            img.save(output_path)
            logger.debug(f"Successfully saved image to {output_path}")
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            raise

    def extract_and_replace_images(
        self,
        markdown_content: str,
        source: str
    ) -> Tuple[str, List[Document]]:
        """
        Extract base64 images from markdown content, save them locally, 
        and replace them with a placeholder in the text.

        :param markdown_content: The raw markdown text.
        :param source: The source filename for metadata tracking.
        :return: A tuple containing (cleaned_text, list_of_image_documents).
        """
        image_docs = []
        
        # Regex to match markdown images: ![alt](data:image/type;base64,data)
        # Group 1: Alt text (lazy match)
        # Group 2: Image type (png, jpeg, etc.)
        # Group 3: Base64 Data
        pattern = r'!\[(.*?)\]\(data:image/(.*?);base64,(.*?)\)'

        def replace_callback(match):
            alt_text = match.group(1)
            img_type = match.group(2).split(';')[0] # Safety split if type has extra params
            base64_data = match.group(3)

            # Generate unique filename using hash
            hash_key = hashlib.md5(base64_data.encode()).hexdigest()
            # Normalize extension
            ext = img_type if img_type in ['png', 'jpg', 'jpeg'] else 'png'
            filename = f"{hash_key}.{ext}"
            image_path = os.path.join(self.images_output_dir, filename)

            # Save image to disk
            self.save_base64_to_image(base64_data, image_path)

            # Create Image Document
            doc = Document(
                page_content = str(image_path),
                metadata = {
                    "source": source,
                    "alt_text": alt_text if alt_text else "image",
                    "embedding_type": "image"
                }
            )
            image_docs.append(doc)

            # Return placeholder for text content
            return "[image]"

        # Perform substitution and extraction in one pass
        cleaned_content = re.sub(pattern, replace_callback, markdown_content, flags = re.DOTALL)
        
        return cleaned_content, image_docs

    @staticmethod
    def convert_html_table_to_markdown(html_table: str) -> str:
        """
        Convert a single HTML table string to Markdown format.

        :param html_table: String containing <table>...</table>.
        :return: Markdown formatted table string.
        """
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return html_table
            
            markdown_lines = []
            headers = []
            rows = []

            # 1. Extract Headers
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = [th.get_text(strip = True) for th in header_row.find_all(['th', 'td'])]
            
            # Fallback: check first row of tbody or table if no thead
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    # Treat as header if it contains <th> or if it's the very first row
                    cells = first_row.find_all(['th', 'td'])
                    if any(c.name == 'th' for c in cells):
                        headers = [c.get_text(strip = True) for c in cells]

            # 2. Extract Data Rows
            # If we found headers in the first row manually, skip it in data loop
            all_rows = table.find_all('tr')
            start_index = 1 if (headers and all_rows and all_rows[0].find('th')) else 0
            
            # If explicit thead existed, we iterate tbody usually, but find_all('tr') is safer for malformed html
            # Let's rely on logic: if a row matches the extracted header row object, skip it.
            
            for tr in all_rows:
                # Basic dedup: if this tr is inside thead, skip (already handled)
                if tr.parent.name == 'thead':
                    continue
                
                # If we manually grabbed headers from the first tr (which wasn't in thead), skip it
                if headers and not thead and tr == all_rows[0]:
                    continue

                row_data = [td.get_text(strip = True) for td in tr.find_all(['td', 'th'])]
                if any(row_data): # Only add non-empty rows
                    rows.append(row_data)
            
            # 3. Build Markdown
            if headers:
                markdown_lines.append('| ' + ' | '.join(headers) + ' |')
                markdown_lines.append('|' + '|'.join(['---' for _ in headers]) + '|')
            
            for row in rows:
                # Handle column mismatch
                if headers:
                    while len(row) < len(headers):
                        row.append('')
                    row = row[:len(headers)]
                markdown_lines.append('| ' + ' | '.join(row) + ' |')
            
            if not markdown_lines:
                return html_table
            
            return '\n' + '\n'.join(markdown_lines) + '\n'
            
        except Exception as e:
            logger.error(f"HTML Table conversion failed: {e}")
            return html_table

    def convert_html_to_markdown(self, text: str) -> str:
        """
        Scan text for HTML tables and convert them to Markdown.

        :param text: Text containing potential HTML tables.
        :return: Text with Markdown tables.
        """
        def replace_table(match):
            return self.convert_html_table_to_markdown(match.group(0))
        
        # Regex matches <table>...</table> across newlines
        converted_text = re.sub(
            r'<table>.*?</table>', 
            replace_table, 
            text, 
            flags = re.DOTALL | re.IGNORECASE
        )
        return converted_text

    def process_md_file(self, md_file: str) -> List[Document]:
        """
        Process a single markdown file: Convert Tables -> Split Headers -> Extract Images -> Semantic Split.

        :param md_file: Path to the markdown file.
        :return: List of processed Documents.
        """
        try:
            with open(md_file, 'r', encoding = 'utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"File not found: {md_file}")
            return []

        logger.info(f"Processing file: {md_file} (Length: {len(content)})")
        
        # 1. Pre-process: Convert HTML tables
        content = self.convert_html_to_markdown(content)

        # 2. Structural Split: Split by Markdown headers
        # Note: This might split text mid-sentence if headers are used improperly, but standard for MD.
        split_documents = self.header_splitter.split_text(content)
        
        documents = []
        for doc in split_documents:
            # 3. Image Extraction & Cleaning
            # Combine extraction and cleaning into one pass to avoid regex mismatch errors
            cleaned_content, image_docs = self.extract_and_replace_images(
                doc.page_content, 
                md_file
            )
            
            # 4. Handle Text Content
            if cleaned_content.strip():
                # Update document with cleaned text and mark as text
                text_doc = Document(
                    page_content = cleaned_content, 
                    metadata = doc.metadata.copy()
                )
                text_doc.metadata['embedding_type'] = 'text'
                documents.append(text_doc)
            
            # 5. Handle Image Content
            # Image docs already have correct metadata from extract_and_replace_images
            # We just need to merge any header metadata if needed, 
            # though images usually belong to the section they are in.
            for img_doc in image_docs:
                # Merge header metadata from the parent doc into the image doc
                img_doc.metadata.update({k: v for k, v in doc.metadata.items() if k not in img_doc.metadata})
                documents.append(img_doc)
        
        # 6. Semantic Split: Handle over-length text chunks
        final_docs = []
        for d in documents:
            if d.metadata.get('embedding_type') == 'text' and len(d.page_content) > self.text_chunk_size:
                final_docs.extend(self.recursive_splitter.split_documents([d]))
            else:
                final_docs.append(d)

        return final_docs
    
    def add_title_hierarchy(
        self,
        documents: List[Document],
        source_filename: str
    ) -> List[Document]:
        """
        Post-process documents to fill in missing hierarchical headers across file boundaries.
        Essential when processing paginated markdown files (e.g. page_01.md, page_02.md).

        :param documents: Flat list of all documents from all files.
        :param source_filename: The original source name (e.g. PDF name) to attach.
        :return: List of documents with complete header hierarchy.
        """
        # State tracker for current active headers
        current_titles = {1: "", 2: "", 3: ""}
        processed_docs = []

        for doc in documents:
            new_metadata = doc.metadata.copy()
            new_metadata['source'] = source_filename

            # Update state: If a doc has a header, it updates the current context
            # and invalidates all lower-level headers.
            for level in range(1, 4):
                header_key = f'header_{level}'
                if header_key in new_metadata:
                    current_titles[level] = new_metadata[header_key]
                    # Reset lower levels because a new H(n) starts a fresh section for H(n+1)
                    for lower_level in range(level + 1, 4):
                        current_titles[lower_level] = ""

            # Fill missing headers: If a doc lacks a header, inherit from current context
            for level in range(1, 4):
                header_key = f'header_{level}'
                if header_key not in new_metadata and current_titles[level]:
                    new_metadata[header_key] = current_titles[level]
            
            processed_docs.append(Document(page_content = doc.page_content, metadata = new_metadata))

        return processed_docs
    
    def process_md_dir(
        self,
        md_dir: str,
        source_filename: str
    ) -> List[Document]:
        """
        Process all markdown files in a directory in order.

        :param md_dir: Directory containing .md files.
        :param source_filename: Logical source name for the collection.
        :return: List of final processed documents.
        """
        md_files = get_sorted_md_files(md_dir)
        all_documents = []
        
        logger.info(f"Found {len(md_files)} files in {md_dir}")

        for md_file in md_files:
            all_documents.extend(self.process_md_file(md_file))
            
        return self.add_title_hierarchy(all_documents, source_filename)

if __name__ == "__main__":
    # Configure logging strictly inside main as requested
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    # Use argparse for script execution
    parser = argparse.ArgumentParser(description = "Test Markdown Splitter Module")
    parser.add_argument("--test_dir", type = str, default = "examples/md_splitter_test/md_files/", help = "Directory of MD files")
    parser.add_argument("--output_dir", type = str, default = "examples/md_splitter_test/output/images", help = "Image output directory")
    parser.add_argument("--source_name", type = str, default = "LLM.md", help = "Source filename for metadata")
    
    args = parser.parse_args()

    # Execution
    splitter = MarkdownDirSplitter(images_output_dir = args.output_dir)
    documents = splitter.process_md_dir(args.test_dir, source_filename = args.source_name)

    # Visualization of results
    for i, doc in enumerate(documents[:5]): # Show first 5 for brevity
        logger.info(f"Doc {i+1} [{doc.metadata.get('embedding_type', 'unknown')}]")
        logger.info("-" * 50)
        content_preview = doc.page_content[:50].replace('\n', ' ') + "..."
        logger.info(f"Content: {content_preview}")
        logger.info(f"Headers: H1={doc.metadata.get('header_1', 'N/A')} | H2={doc.metadata.get('header_2', 'N/A')}")
        logger.info("-" * 50)