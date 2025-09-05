# app.py

import os
import time
from pathlib import Path
from dotenv import load_dotenv 
from utils import load_and_chunk_document, create_vector_store, create_conversational_rag_chain

# Find the project root directory by going up one level from the script's location
# __file__ is the path to the current script (app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- CONFIGURATION ---
PDF_FILE_PATH = PROJECT_ROOT / "data" / "distributed-and-cloud-computing-from-parallel-processing-to-the-internet-of-things.pdf"

# Path to where you'll save the vector store
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"


# The local LLM model you're using
LLM_MODEL = "phi3:mini"

# app.py (continued)

def initialize_vector_store():
    """Checks if the vector store exists, creates it if it doesn't."""
    
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"✅ Loading existing vector store from '{VECTOR_STORE_PATH}'...")
        # To load a FAISS vector store, you need the same embedding model it was created with
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True) # FAISS.save_local creates two files: index.faiss (the raw numerical index) and index.pkl (a pickle file mapping the index's internal IDs back to the LangChain Document objects). The load_local function reconstructs the FAISS index in memory and re-links it to the document content. You must provide the exact same embedding model it was created with, because the FAISS object itself doesn't store the model. It needs it to embed new queries for searching. The allow_dangerous_deserialization=True is required because loading pickle files can be a security risk if the file is from an untrusted source, so LangChain makes you explicitly enable it.
        print("✅ Vector store loaded successfully.")
    else:
        print("Vector store not found. Creating a new one...")
        print("-" * 50)
        
        # 1. Load and chunk the document
        doc_chunks = load_and_chunk_document(PDF_FILE_PATH)
        
        # 2. Create the vector store
        vector_store = create_vector_store(doc_chunks)
        
        # 3. Save the vector store to disk for future use
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"✅ New vector store created and saved to '{VECTOR_STORE_PATH}'.")
        print("-" * 50)
        
    return vector_store

# app.py (continued)

def main():
    """The main function to run the study buddy application."""

    # Load environment variables from the .env file at the project root
    load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
    
    # Securely get the Cohere API key from the environment
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key or cohere_api_key == "YOUR_TRIAL_API_KEY_FROM_COHERE_WEBSITE_GOES_HERE":
        print("⚠️  Warning: COHERE_API_KEY not found in .env file. Re-ranker will be disabled.")
        print("Please get a free key from https://cohere.com/ and add it to your .env file.")
        cohere_api_key = None # Ensure it's None if not set

    # Initialize the vector store (either load or create)
    try:
        vector_db = initialize_vector_store()
    except FileNotFoundError:
        print(f"\n❌ ERROR: The file '{PDF_FILE_PATH}' was not found.")
        print("Please update the 'PDF_FILE_PATH' variable in app.py with the correct path.")
        return
    except Exception as e:
        print(f"\n❌ An error occurred during initialization: {e}")
        return

    # Setup the RAG pipeline using the vector store
    study_buddy_chain = create_conversational_rag_chain(vector_db, LLM_MODEL, cohere_api_key)
    
    print("\n🚀 Study Buddy is ready! Type 'exit' to quit.")
    print("-" * 50)

    chat_history = []

    # Interactive loop
    while True:
        try:
            user_query = input("\nAsk a question: ")
            if user_query.lower() == 'exit':
                print("Goodbye! Happy studying.")
                break
            
            start_time = time.time()
            
            # Get the answer from the RAG chain
            response = study_buddy_chain.invoke(
                {"question": user_query, "chat_history": chat_history}
            )
            chat_history.append((user_query, response['answer']))
            
            end_time = time.time()
            
            print("\n--- Answer ---")
            print(response['answer'])
            print("--------------")
            print(f"(Responded in {end_time - start_time:.2f} seconds)")

        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy studying.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    main()