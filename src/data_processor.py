# In src/data_processor.py

import os
import fitz  # This is PyMuPDF
from pathlib import Path
from tqdm import tqdm
import pickle
import base64
import uuid
from unstructured.partition.pdf import partition_pdf

# --- 1. CONFIGURATION ---
# This block sets up all the necessary paths for our project.
# Path(__file__) gets the path of the current script (data_processor.py).
# .resolve() makes it an absolute path.
# .parent.parent navigates up two levels to get to the project's root folder ('CogniText').
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# We construct the full path to our textbook PDF.
PDF_FILE_PATH = PROJECT_ROOT / "data" / "distributed-and-cloud-computing-from-parallel-processing-to-the-internet-of-things.pdf"

# This is the directory where all our processed output will be stored.
OUTPUT_DIR = PROJECT_ROOT / "processed_data"

# --- 2. THE CORE PDF PARTITIONING FUNCTION ---

def process_pdf_and_extract_elements(pdf_path, output_dir):
    """
    Uses the 'unstructured' library to partition a PDF into a list of elements.
    It saves extracted images directly to a specified folder and returns lists
    of text and table elements for further processing.
    """
    print(f"Starting PDF processing for: {pdf_path.name}")
    
    # We define a specific sub-directory within our output folder to store the images.
    image_output_dir = output_dir / "images"
    # This creates the directory if it doesn't already exist.
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    # This is the core function call from the 'unstructured' library. It's incredibly powerful.
    raw_pdf_elements = partition_pdf(
        filename=str(pdf_path),
        # strategy="hi_res" tells unstructured to use its high-resolution,
        # computer-vision-based model (like YOLO) to analyze the PDF's layout visually.
        # This allows it to understand columns, headers, footers, and separate images from text.
        strategy="hi_res",
        
        # We explicitly tell it to look for and extract tables.
        infer_table_structure=True,
        
        # This tells it to extract images.
        extract_images_in_pdf=True,
        # This tells unstructured to embed the image data directly into the element object
        # instead of saving it as a side-effect.
        extract_image_block_to_payload=True,
        # And this is where it will save the extracted image files.
        # image_output_dir_path=str(image_output_dir),
    )

    # --- 3. CATEGORIZE THE EXTRACTED ELEMENTS ---
    # The output of partition_pdf is a single list of mixed elements. We need to separate them.
    texts = []
    tables = []

    # We define a tuple of the text-based element types we explicitly want to capture.
    # This acts as a "whitelist". We are only interested in these types for our text data.
    TEXT_ELEMENT_TYPES = (
        "unstructured.documents.elements.NarrativeText",
        "unstructured.documents.elements.Title",
        "unstructured.documents.elements.ListItem"
    )
    
    print("Categorizing extracted elements...")
    # tqdm gives us a nice progress bar, which is helpful for long processes.
    for element in tqdm(raw_pdf_elements, desc="Processing Elements"):
        element_type = str(type(element))
        # We check the type of each element to see if it's a table.
        if 'unstructured.documents.elements.Table' in element_type:
            # For tables, we store both the plain text representation and the HTML representation.
            # The HTML is often better for an LLM to understand the table's structure.
            # We also save the page number from the metadata.
            tables.append({
                "text": str(element),
                "html": element.metadata.text_as_html,
                "source_page": element.metadata.page_number
            })
        # CompositeElement usually represents a block of text, like a paragraph.
        # Second, we check if the element's type is in our whitelist of text types.
        # This is much safer than a catch-all 'else'.
        elif any(t in element_type for t in TEXT_ELEMENT_TYPES):
            texts.append({
                "text": str(element),
                "source_page": element.metadata.page_number
            })
        # If an element is neither a Table nor one of our desired text types (e.g., it's an
        # Image, Header, Footer, etc.), this loop will simply ignore it and move on.
        # This is the correct, robust behavior.
        # Images are automatically saved to the folder, so we don't need to handle them here.
        elif 'unstructured.documents.elements.Image' in element_type:
            # The image data is now in the element's metadata payload.
            image_data = element.metadata.image_base64
            image_format = element.metadata.image_mime_type.split('/')[-1] # e.g., 'jpeg'
            
            # Create a unique filename for the image.
            image_filename = f"page_{element.metadata.page_number}_img_{uuid.uuid4()}.{image_format}"
            image_path = image_output_dir / image_filename
            
            # Decode the base64 string back into binary data and save the image.
            with open(image_path, "wb") as img_file:
                img_file.write(base64.b64decode(image_data))

    print(f"Found {len(texts)} text elements and {len(tables)} table elements.")
    return texts, tables

# --- 4. MAIN EXECUTION BLOCK ---

def main():
    """
    This is the main function that runs when you execute the script.
    It orchestrates the entire data preparation process.
    """
    print("--- Starting Data Processing for CogniText ---")
    
    # Create the main output directory if it's not there.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Call our function to partition the PDF.
    texts, tables = process_pdf_and_extract_elements(PDF_FILE_PATH, OUTPUT_DIR)

    # Save the processed lists to disk using pickle.
    # Pickling is a way to serialize Python objects into a byte stream, so we can
    # save them to a file and load them back later. This is much faster than
    # reprocessing the PDF every time our main app starts.
    with open(OUTPUT_DIR / "texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    with open(OUTPUT_DIR / "tables.pkl", "wb") as f:
        pickle.dump(tables, f)

    print("\n✅ Data processing complete!")
    print(f"Processed text and tables saved in: {OUTPUT_DIR}")
    print(f"Extracted images are saved in: {OUTPUT_DIR / 'images'}")

if __name__ == "__main__":
    # This standard Python construct ensures that the main() function is
    # only called when the script is executed directly.
    main()