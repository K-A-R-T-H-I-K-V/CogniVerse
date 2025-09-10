import os
import pickle
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from IPython.display import display
import uuid

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path.cwd().parent
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
ID_KEY = "doc_id"

def main():
    if not VECTOR_STORE_PATH.exists():
        print(f"❌ Vector store not found at '{VECTOR_STORE_PATH}'.")
        return

    print("--- Initializing Inspector Tool ---")
    
    vectorstore = Chroma(
        collection_name="cogniverse-final-v7",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        persist_directory=str(VECTOR_STORE_PATH)
    )
    print(f"✅ Vector store loaded with {vectorstore._collection.count()} summaries.")

    # --- Load all original documents into a searchable dictionary ---
    print("\nLoading original processed data...")
    docstore = {}
    with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f: texts = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f: tables = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "image_paths.pkl", "rb") as f: image_paths = pickle.load(f)
    
    all_docs_with_ids = []
    # Recreate the exact same UUIDs and data order as the app
    all_text_docs = texts + tables
    text_doc_ids = [str(uuid.uuid4()) for _ in all_text_docs] # This is a placeholder, we need the real IDs
    
    # The only truly reliable way is to query ALL documents from Chroma
    all_db_data = vectorstore.get(include=["metadatas", "documents"])
    summary_to_id_map = {doc: meta[ID_KEY] for doc, meta in zip(all_db_data["documents"], all_db_data["metadatas"])}

    # Rebuild the docstore from our original files
    # This is complex because we didn't save the UUIDs. The main app does this better.
    # This inspector will focus on showing the summaries, which is the key part.
    
    while True:
        query = input("\nEnter a query to test (or 'exit' to quit): ")
        if query.lower() == 'exit': break
        
        print("\n--- Performing direct similarity search... ---")
        retrieved_summaries = vectorstore.similarity_search_with_score(query, k=5)
        
        if not retrieved_summaries:
            print("\n❌ Retrieval FAILED. No relevant summaries found.")
        else:
            print(f"\n✅ Retrieval SUCCESS. Found {len(retrieved_summaries)} relevant summaries:")
            results_data = []
            for i, (doc, score) in enumerate(retrieved_summaries):
                results_data.append({
                    "Result": i+1,
                    "Similarity Score (Lower is Better)": f"{score:.4f}",
                    "Retrieved Summary": doc.page_content
                })
            df = pd.DataFrame(results_data)
            display(df)

if __name__ == "__main__":
    main()