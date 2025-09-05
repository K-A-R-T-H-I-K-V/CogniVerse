# src/debug_retrieval.py

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever # The orchestrator for the retrieve-then-rerank process
from langchain_cohere import CohereRerank # The intelligent re-ranking model from Cohere

def main():
    """
    A script to debug the retrieval process, now including a re-ranking step
    to show how it improves context quality.
    """
    # --- Same path logic as app.py ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
    cohere_api_key = os.getenv("COHERE_API_KEY")
    VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"

    print("--- Retrieval Debugger with Re-Ranking ---")
    
    if not cohere_api_key or "YOUR_TRIAL_API_KEY" in cohere_api_key:
        print("❌ Error: Cohere API Key is missing or invalid in your .env file.")
        print("Please get a free key from https://cohere.com/ and add it to .env to run this debugger.")
        return
    
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"❌ Error: Vector store not found at '{VECTOR_STORE_PATH}'")
        print("Please run 'python src/app.py' at least once to create it.")
        return

    # 1. Load the existing vector store
    print("Loading vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(
            str(VECTOR_STORE_PATH), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded.")
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return

    # 1. Base Retriever: We create a standard retriever from our vector store.
    base_retriever = vector_store.as_retriever(search_kwargs={'k': 10}) # Get top 10 chunks
    # 2. Re-ranker Model: We initialize the Cohere Rerank model.
    # top_n=3 tells the model: "Out of all the documents you receive,
    # only return the top 3 most relevant ones."
    cohere_rerank = CohereRerank(
            cohere_api_key=cohere_api_key, top_n=3, model="rerank-english-v3.0"
        ) # Re-rank and keep
    # 3. Compression Retriever: This is the orchestrator. It connects the two parts.
    # It takes the 'base_retriever' (the fast search) and the 'base_compressor'
    # (the intelligent re-ranker) and creates a new retriever that automatically
    # performs the two-step "retrieve-then-rerank" process.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank, base_retriever=base_retriever
    )
    final_retriever = compression_retriever
    
    # 3. Interactive loop to test retrieval
    while True:
        try:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            print(f"\n🔍 Retrieving and re-ranking chunks for query: '{query}'")
            # This single .invoke() call now triggers the entire chain:
            # 1. 'final_retriever' calls 'base_retriever' to get 10 chunks.
            # 2. It then sends those 10 chunks to the 'cohere_rerank' model.
            # 3. The re-ranker returns the best 3 chunks, which are stored here.
            reranked_chunks = final_retriever.invoke(query) # This is the retrieval step. The retriever takes your text query, converts it into an embedding (a vector), and then searches the FAISS index for the 4 vectors (chunks) that are most similar to your query's vector.
            
            print(f"\nFound {len(reranked_chunks)} re-ranked chunks:\n")
            
            for i, chunk in enumerate(reranked_chunks):
                # Cohere adds a 'relevance_score' to the metadata of each chunk.
                # We can print this score to see how confident it is about each result.
                relevance_score = chunk.metadata.get('relevance_score', 0)
                source_info = chunk.metadata.get('Header 1', 'N/A')
                print(f"--- Chunk {i+1} (Relevance: {chunk.metadata.get('relevance_score'):.4f}, Source: {source_info}) ---")
                print(chunk.page_content)
                print("-" * 20 + "\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()