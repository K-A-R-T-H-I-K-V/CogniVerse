# In src/data_processor.py (FINAL, ULTIMATE HYBRID VERSION)

import pickle
import re
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF is used as a fallback and for page text ground truth
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import MarkdownHeaderTextSplitter
import base64
import uuid

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_FILE_PATH = PROJECT_ROOT / "data" / "cleaned_BCS502_CN_Mod1.pdf" # Make sure this is your new file
OUTPUT_DIR = PROJECT_ROOT / "processed_data"

# --- 2. HYBRID PROCESSING LOGIC ---

def process_pdf_elements(raw_pdf_elements: list, output_dir: Path) -> tuple[list, list, list]:
    """
    Processes a list of Unstructured elements to separate out text, tables, and images.
    Images are saved to disk.
    """
    print("Categorizing elements and extracting images...")
    
    text_elements, tables = [], []
    image_paths = []
    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    for element in tqdm(raw_pdf_elements, desc="Processing elements"):
        if "unstructured.documents.elements.Image" in str(type(element)):
            # Handle images: save them and record their paths
            image_data = element.metadata.image_base64
            image_format = element.metadata.image_mime_type.split('/')[-1]
            image_filename = f"page_{element.metadata.page_number}_img_{uuid.uuid4().hex[:8]}.{image_format}"
            image_path = image_output_dir / image_filename
            with open(image_path, "wb") as img_file:
                img_file.write(base64.b64decode(image_data))
            image_paths.append(str(image_path))
        elif "unstructured.documents.elements.Table" in str(type(element)):
            # Handle tables
            tables.append({
                "html": element.metadata.text_as_html,
                "text": element.text,
                "source_page": element.metadata.page_number
            })
        else:
            # Handle text-based elements (Title, NarrativeText, ListItem, etc.)
            text_elements.append(element)
            
    print(f"✅ Found {len(text_elements)} text elements, {len(tables)} tables, and {len(image_paths)} images.")
    return text_elements, tables, image_paths

def elements_to_markdown(text_elements: list) -> str:
    """Converts a list of Unstructured text elements into a single Markdown string."""
    markdown_text = ""
    for el in text_elements:
        if el.category == "Title":
            if re.match(r'^\d+\.\d+\.\d+', el.text): markdown_text += f"### {el.text}\n\n"
            elif re.match(r'^\d+\.\d+', el.text): markdown_text += f"## {el.text}\n\n"
            else: markdown_text += f"# {el.text}\n\n"
        elif el.category == "ListItem":
            clean_text = re.sub(r'^(?:[\s*•-]+)\s*', '', el.text)
            markdown_text += f"* {clean_text}\n"
        else:
            markdown_text += f"{el.text}\n\n"
    return markdown_text

def chunk_markdown(markdown_text: str, doc_pages: list) -> list[dict]:
    """Splits the Markdown text and re-associates chunks with their correct page number."""
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_chunks = md_splitter.split_text(markdown_text)
    final_chunks = []
    for chunk in tqdm(md_chunks, desc="Associating page numbers"):
        first_line = chunk.page_content.split('\n', 1)[0]
        cleaned_first_line = re.sub(r'^[#*\s]+', '', first_line).strip()
        found_page = "N/A"
        if cleaned_first_line:
            for page_num, page_text in doc_pages:
                if cleaned_first_line in page_text:
                    found_page = page_num
                    break
        final_chunks.append({"text": chunk.page_content, "source_page": found_page})
    return final_chunks

# --- 3. MAIN EXECUTION BLOCK ---
def main():
    print("--- Starting Data Processing (Ultimate Unstructured Hybrid Method) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Partition PDF into all elements using Unstructured. This is the key step.
    raw_pdf_elements = partition_pdf(
        filename=str(PDF_FILE_PATH),
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_to_payload=True, # Use the reliable payload method
    )

    # 2. Separate elements into text, tables, and save images
    text_elements, tables, image_paths = process_pdf_elements(raw_pdf_elements, OUTPUT_DIR)
    with open(OUTPUT_DIR / "tables.pkl", "wb") as f: pickle.dump(tables, f)
    with open(OUTPUT_DIR / "image_paths.pkl", "wb") as f: pickle.dump(image_paths, f)
    
    # 3. Get raw page text from PyMuPDF for accurate page number association
    doc = fitz.open(PDF_FILE_PATH)
    doc_pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    doc.close()

    # 4. Convert text elements to Markdown, then chunk them
    markdown_content = elements_to_markdown(text_elements)
    text_chunks = chunk_markdown(markdown_content, doc_pages)
    with open(OUTPUT_DIR / "texts.pkl", "wb") as f: pickle.dump(text_chunks, f)

    print("\n✅ Ultimate hybrid data processing complete!")

if __name__ == "__main__":
    main()