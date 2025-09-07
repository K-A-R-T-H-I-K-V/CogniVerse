# In src/data_processor.py (FINAL Unstructured + MarkdownSplitter VERSION)

import pickle
import re
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF is still the most reliable for images
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import MarkdownHeaderTextSplitter

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_FILE_PATH = PROJECT_ROOT / "data" / "distributed-and-cloud-computing-from-parallel-processing-to-the-internet-of-things.pdf"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"

# --- 2. NEW HYBRID PROCESSING LOGIC ---

def elements_to_markdown(raw_pdf_elements: list) -> str:
    """
    Converts a list of Unstructured elements into a single Markdown string.
    This is the core of your brilliant idea.
    """
    print("Converting Unstructured elements to Markdown...")
    markdown_text = ""
    for element in tqdm(raw_pdf_elements, desc="Converting elements"):
        category = element.metadata.category
        text = element.text
        
        if category == "Title":
            # Use '#' for main titles/headings
            markdown_text += f"# {text}\n\n"
        elif category == "ListItem":
            # Use '*' for list items, ensuring clean formatting
            # This regex removes any stray bullets unstructured might have added
            clean_text = re.sub(r'^(?:[\s*•-]+)\s*', '', text)
            markdown_text += f"* {clean_text}\n"
        elif category == "NarrativeText":
            # Regular text paragraphs
            markdown_text += f"{text}\n\n"
        else:
            # Catch-all for other element types
            markdown_text += f"{text}\n\n"
            
    print("✅ Element to Markdown conversion complete.")
    return markdown_text

def chunk_markdown_with_page_info(markdown_text: str, doc_pages: list) -> list[dict]:
    """
    Splits the Markdown text and re-associates chunks with their correct page number.
    """
    print("Chunking Markdown and associating page numbers...")
    headers_to_split_on = [
        ("#", "Header 1"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    
    md_chunks = markdown_splitter.split_text(markdown_text)
    
    # Associate chunks with the page number of their first line.
    # This is a robust way to handle page numbers.
    final_chunks = []
    for chunk in tqdm(md_chunks, desc="Associating page numbers"):
        first_line = chunk.page_content.split('\n', 1)[0]
        found_page = "N/A"
        
        # Search through the original page texts to find the page number
        for page_num, page_text in doc_pages:
            if first_line in page_text:
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
    # This function remains the same as the previous version.
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
    print("--- Starting Data Processing (Unstructured + Markdown Hybrid Method) ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Partition PDF into elements using Unstructured
    raw_pdf_elements = partition_pdf(
        filename=str(PDF_FILE_PATH),
        strategy="hi_res",
        infer_table_structure=True,
    )

    # 2. Convert elements to a single Markdown string
    markdown_content = elements_to_markdown(raw_pdf_elements)
    
    # 3. For accurate page number association, we'll get raw page text from PyMuPDF
    doc = fitz.open(PDF_FILE_PATH)
    doc_pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    doc.close()

    # 4. Chunk the markdown and add page numbers
    text_chunks = chunk_markdown_with_page_info(markdown_content, doc_pages)
    with open(OUTPUT_DIR / "texts.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

    # 5. Extract images using the reliable PyMuPDF method
    image_paths = extract_images(PDF_FILE_PATH, OUTPUT_DIR)
    with open(OUTPUT_DIR / "image_paths.pkl", "wb") as f:
        pickle.dump(image_paths, f)

    # 6. Create empty tables.pkl for compatibility
    with open(OUTPUT_DIR / "tables.pkl", "wb") as f:
        pickle.dump([], f)

    print("\n✅ Hybrid data processing complete!")
    print(f"Processed data saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

