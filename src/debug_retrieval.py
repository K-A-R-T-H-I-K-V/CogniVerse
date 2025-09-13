import pickle
from pathlib import Path
import textwrap

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

def view_chunks_for_page(page_number, texts_data):
    """
    Finds and prints all text chunks that were extracted from a specific page.
    """
    print("-" * 80)
    print(f"🔎 Searching for text chunks from PAGE {page_number}...")
    print("-" * 80)

    found_chunks = False
    for i, chunk in enumerate(texts_data):
        if chunk.get('source_page') == page_number:
            found_chunks = True
            print(f"\n--- Chunk {i+1} on Page {page_number} ---")
            # textwrap helps print long text neatly
            wrapped_text = textwrap.fill(chunk['text'], width=80)
            print(wrapped_text)
            print("-" * 25)

    if not found_chunks:
        print(f"❌ No text chunks were found in the database for page {page_number}.")

def main():
    """
    Main function to load data and run the debug view.
    """
    try:
        with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f:
            texts = pickle.load(f)
        print(f"✅ Successfully loaded {len(texts)} text chunks from texts.pkl.")

        # --- WHICH PAGE DO YOU WANT TO INSPECT? ---
        # We know the problem is on page 215 from your screenshot.
        view_chunks_for_page(215, texts)

    except FileNotFoundError:
        print(f"❌ Error: Could not find 'texts.pkl' in the '{PROCESSED_DATA_DIR}' directory.")
        print("Please make sure you have run the data_processor.py script first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

