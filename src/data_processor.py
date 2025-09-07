# In src/data_processor.py (FINAL VERSION with all fixes)

import pickle
import re
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF for reliable page text and images
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import MarkdownHeaderTextSplitter

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_FILE_PATH = PROJECT_ROOT / "data" / "distributed-and-cloud-computing-from-parallel-processing-to-the-internet-of-things.pdf"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"

# --- 2. NEW HYBRID PROCESSING LOGIC ---

def elements_to_markdown(raw_pdf_elements: list) -> tuple[str, list]:
    """
    Converts a list of Unstructured elements into a single Markdown string
    and separates out the tables.
    """
    print("Converting Unstructured elements to Markdown and separating tables...")
    markdown_text = ""
    tables = []
    
    for element in tqdm(raw_pdf_elements, desc="Converting elements"):
        category = element.category
        text = element.text
        
        if category == "Table":
            # Handle tables separately, extracting HTML if available
            table_html = element.metadata.text_as_html
            if table_html:
                tables.append({
                    "html": table_html,
                    "text": text,
                    "source_page": element.metadata.page_number
                })
            continue # Don't add table text to the main markdown content
            
        elif category == "Title":
            # FIX 1: Use regex to detect structure and apply correct heading level
            if re.match(r'^\d+\.\d+\.\d+', text): # e.g., 3.1.1
                markdown_text += f"### {text}\n\n"
            elif re.match(r'^\d+\.\d+', text): # e.g., 3.1 or 4.1
                markdown_text += f"## {text}\n\n"
            else: # Main chapter titles
                markdown_text += f"# {text}\n\n"
        elif category == "ListItem":
            clean_text = re.sub(r'^(?:[\s*•-]+)\s*', '', text)
            markdown_text += f"* {clean_text}\n"
        else: # NarrativeText and others
            markdown_text += f"{text}\n\n"
            
    print("✅ Element processing complete.")
    return markdown_text, tables

def chunk_markdown_with_page_info(markdown_text: str, doc_pages: list) -> list[dict]:
    """
    Splits the Markdown text and re-associates chunks with their correct page number.
    """
    print("Chunking Markdown and associating page numbers...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    
    md_chunks = markdown_splitter.split_text(markdown_text)
    
    final_chunks = []
    for chunk in tqdm(md_chunks, desc="Associating page numbers"):
        first_line = chunk.page_content.split('\n', 1)[0]
        
        # FIX 2: Clean the markdown from the first line before comparing
        cleaned_first_line = re.sub(r'^[#*\s]+', '', first_line).strip()
        
        found_page = "N/A"
        if cleaned_first_line: # Ensure we don't search for empty strings
            for page_num, page_text in doc_pages:
                if cleaned_first_line in page_text:
                    found_page = page_num
                    break
        
        final_chunks.append({
            "text": chunk.page_content,
            "source_page": found_page
        })
        
    print(f"✅ Successfully created and processed {len(final_chunks)} chunks.")
    return final_chunks

def extract_images(pdf_path, output_dir):
    """Extracts images using PyMuPDF (most reliable method)."""
    # This function remains the same.
    print("Extracting images...")
    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in tqdm(range(len(doc)), desc="Extracting Images"):
        page = doc.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page_num + 1}_img_{img_index}.{image_ext}"
            image_path = image_output_dir / image_filename
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(str(image_path))
    print(f"✅ Successfully extracted {len(image_paths)} images.")
    return image_paths

# --- 3. MAIN EXECUTION BLOCK ---
def main():
    print("--- Starting Data Processing (FINAL HYBRID Method) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Partition PDF into elements using Unstructured
    raw_pdf_elements = partition_pdf(
        filename=str(PDF_FILE_PATH),
        strategy="hi_res",
        infer_table_structure=True,
    )

    # 2. Convert elements to Markdown and extract tables
    markdown_content, tables = elements_to_markdown(raw_pdf_elements)
    # FIX 3: Save the extracted tables
    with open(OUTPUT_DIR / "tables.pkl", "wb") as f:
        pickle.dump(tables, f)

    # 3. Get raw page text from PyMuPDF for accurate page number association
    doc = fitz.open(PDF_FILE_PATH)
    doc_pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    doc.close()

    # 4. Chunk the markdown and add page numbers
    text_chunks = chunk_markdown_with_page_info(markdown_content, doc_pages)
    with open(OUTPUT_DIR / "texts.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

    # 5. Extract images
    image_paths = extract_images(PDF_FILE_PATH, OUTPUT_DIR)
    with open(OUTPUT_DIR / "image_paths.pkl", "wb") as f:
        pickle.dump(image_paths, f)

    print(f"\n✅ All processing complete! Found {len(tables)} tables.")

if __name__ == "__main__":
    main()