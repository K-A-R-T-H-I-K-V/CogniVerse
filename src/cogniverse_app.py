# In src/cogniverse_app.py

import os
import uuid
import base64
import pickle
from pathlib import Path
from dotenv import load_dotenv
import time

# --- Core LangChain Imports ---
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Specific LangChain Component Imports ---
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_cohere import CohereRerank
from langchain_google_genai import ChatGoogleGenerativeAI

from tqdm import tqdm

# --- 1. CONFIGURATION and LOADING ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"

# We use the Gemini API for the one-time, heavy-lifting summary generation for images.
SUMMARY_MODEL_API = "gemini-1.5-flash"
# We use local models for the fast, interactive parts of the application.
FINAL_RESPONSE_MODEL_LOCAL = "llava"
QUESTION_CONDENSING_MODEL_LOCAL = "phi3:mini"

# --- 2. HELPER FUNCTIONS ---
def image_to_base64(image_path):
    """Converts an image file to a base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def generate_image_summaries_in_batch(llm, image_batch_b64):
    """
    Generates summaries for a batch of images in a single, efficient API call.
    """
    if not image_batch_b64:
        return []
    
    # Construct the multimodal prompt with text instructions and multiple images.
    prompt_content = [
        {"type": "text", "text": "You will be given a list of images from a textbook. Provide a concise, one-sentence summary for EACH image, in order. The summary should capture the main keywords and concepts and will be used for a search index. Output EACH summary on a new line. Do not add any other text or numbering."}
    ]
    for b64 in image_batch_b64:
        prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"})

    try:
        msg = llm.invoke([HumanMessage(content=prompt_content)])
        summaries = [s.strip() for s in msg.content.split('\n') if s.strip()]

        # This is a crucial safety check. If the LLM gets confused and doesn't return
        # the correct number of summaries, we must handle it gracefully.
        if len(summaries) != len(image_batch_b64):
            print(f"\nWarning: Batch image summary count mismatch. Expected {len(image_batch_b64)}, got {len(summaries)}. Adding placeholders.")
            # We add placeholder summaries to ensure the rest of the program doesn't crash.
            summaries.extend(["Image summary generation failed for this item."] * (len(image_batch_b64) - len(summaries)))
        return summaries
    except Exception as e:
        print(f"\nAn error occurred during batch image summarization: {e}")
        # Return placeholders if the entire API call fails.
        return ["Image summary generation failed due to API error."] * len(image_batch_b64)

def format_docs_for_display(docs, image_paths):
    """A utility function to pretty-print the retrieved sources for the user."""
    formatted_string = ""
    text_docs = [doc for doc in docs if not doc.metadata.get('is_image', False)]
    for i, doc in enumerate(text_docs):
        source_page = doc.metadata.get('source_page', 'N/A')
        formatted_string += f"\n--- Retrieved Text/Table {i+1} (Source: Page {source_page}) ---\n"
        formatted_string += doc.page_content[:200] + "...\n"
    if image_paths:
        formatted_string += "\n--- Retrieved Images ---"
        for path in image_paths:
            formatted_string += f"\n- {Path(path).name}"
    return formatted_string

# --- 3. MAIN APPLICATION LOGIC ---

def main():
    print("--- Initializing CogniVerse Multimodal RAG Application ---")

    try:
        with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f:
            texts = pickle.load(f)
        with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f:
            tables = pickle.load(f)
        with open(PROCESSED_DATA_DIR / "image_paths.pkl", "rb") as f:
            image_paths = pickle.load(f)
    except FileNotFoundError:
        print("\n❌ Pre-processed data not found. Please run `python src/data_processor.py` first.")
        return

    # --- Step B: Initialize LLMs ---
    if not GOOGLE_API_KEY or "YOUR_KEY" in GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found or is a placeholder in .env file.")
        return
        
    summary_llm_api = ChatGoogleGenerativeAI(model=SUMMARY_MODEL_API, google_api_key=GOOGLE_API_KEY, temperature=0)
    final_rag_llm_local = OllamaLLM(model=FINAL_RESPONSE_MODEL_LOCAL, temperature=0.1)
    condense_llm_local = OllamaLLM(model=QUESTION_CONDENSING_MODEL_LOCAL, temperature=0)

    # --- Step C: Setup the Multi-Vector Retriever ---
    print("Setting up the Multi-Vector Retriever...")
    vectorstore = Chroma(
        collection_name="cogniverse_final_architecture_v6", # New collection name for a clean build
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=str(VECTOR_STORE_PATH)
    )
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    # --- Step D: Populate the Retriever ---
    if not vectorstore.get()['ids']:
        print("Vector store is empty. Populating with fast hybrid strategy...")
        
        # --- Text and Table Processing (Instant) ---
        all_docs = [Document(page_content=t['text'], metadata={'source_page': t['source_page']}) for t in texts]
        all_docs.extend([Document(page_content=t['html'], metadata={'source_page': t['source_page']}) for t in tables])
        doc_ids = [str(uuid.uuid4()) for _ in all_docs]
        retriever.docstore.mset(list(zip(doc_ids, all_docs)))

        sub_chunk_docs = []
        for i, doc in enumerate(tqdm(all_docs, desc="Creating Text Sub-Chunks")):
            sub_chunk_docs.append(Document(page_content=doc.page_content[:1024], metadata={id_key: doc_ids[i]}))
        
        BATCH_SIZE = 4000
        for i in tqdm(range(0, len(sub_chunk_docs), BATCH_SIZE), desc="Adding Text Sub-Chunks to ChromaDB"):
            retriever.vectorstore.add_documents(sub_chunk_docs[i:i+BATCH_SIZE])
        
        # --- Image Processing (Fast, via Gemini API Batching) ---
        print("Now, generating summaries for images using the Gemini API. This should take a few minutes.")
        
        valid_image_paths = [p for p in image_paths if image_to_base64(p) is not None]
        image_base64s = [image_to_base64(p) for p in valid_image_paths]
        
        IMAGE_BATCH_SIZE = 10 # Your brilliant insight
        image_summaries = []
        for i in tqdm(range(0, len(image_base64s), IMAGE_BATCH_SIZE), desc="Summarizing Images with Gemini (Batch)"):
            batch_b64 = image_base64s[i:i+IMAGE_BATCH_SIZE]
            image_summaries.extend(generate_image_summaries_in_batch(summary_llm_api, batch_b64))
        
        image_ids = [str(uuid.uuid4()) for _ in valid_image_paths]
        image_docs = [Document(page_content=p, metadata={'is_image': True}) for p in valid_image_paths]
        retriever.docstore.mset(list(zip(image_ids, image_docs)))
        
        summary_docs = [Document(page_content=summary, metadata={id_key: image_ids[i]}) for i, summary in enumerate(image_summaries)]
        retriever.vectorstore.add_documents(summary_docs)
        
        vectorstore.persist()
        print("✅ Retriever fully populated and vector store persisted.")
    else:
        print("✅ Vector store already populated. Loading from disk.")

    # --- Step E: Setup the Re-ranker ---
    re_ranker = None
    if COHERE_API_KEY and "YOUR_TRIAL_API_KEY" not in COHERE_API_KEY:
        print("✅ Cohere re-ranker is enabled.")
        re_ranker = CohereRerank(
            cohere_api_key=COHERE_API_KEY, top_n=4, model="rerank-english-v3.0"
        )
    else:
        print("⚠️ Cohere re-ranker is disabled (API key not provided).")

    # --- Step F: Define the Final Conversational RAG Chain ---
    condense_question_prompt = ChatPromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")
    
    condense_question_chain = condense_question_prompt | condense_llm_local | StrOutputParser()

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_question_chain
        else:
            return input["question"]

    def format_for_final_prompt(docs):
        prompt_content = []
        prompt_content.append({"type": "text", "text": """You are an expert university study buddy. Your primary directive is to act as a tutor and give a justifiable, well-written, and easy-to-read answer based STRICTLY AND ONLY on the provided context, which may include text, tables, and images.

**Instructions:**
1.  **Synthesize, Do Not Just List:** Read all the provided context documents. Weave the information into a **single, cohesive, flowing answer.** Do not list information from different documents separately. Your answer should read like a single, well-written explanation from an expert tutor.
2.  **Analyze and Explain Images/Tables:** If images or tables are present in the context, do not just mention them. **Analyze their content** and explain what they illustrate in relation to the user's question.
3.  **Format for Readability:** Use Markdown for formatting. Use headings and subheadings. **Bold** key terms and definitions. Use bullet points for lists.
4.  **Strictly Adhere to Context:** If the context does not contain enough information to answer the question, you MUST respond with exactly this phrase: "Based on the provided textbook, I cannot answer this question." and not a word more. Do not use any outside knowledge.

--- CONTEXT START ---"""})

        for doc in docs:
            if doc.metadata.get('is_image', False):
                image_base64 = image_to_base64(doc.page_content)
                if image_base64:
                    prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"})
            else:
                source_page = doc.metadata.get('source_page', 'N/A')
                prompt_content.append({"type": "text", "text": f"\n[Text/Table from Page {source_page}]:\n{doc.page_content}"})

        prompt_content.append({"type": "text", "text": "\n--- CONTEXT END ---\n"})
        return prompt_content
    
    chain = (
        RunnablePassthrough.assign(standalone_question=condense_question)
        | RunnablePassthrough.assign(retrieved_docs=lambda x: retriever.get_relevant_documents(x["standalone_question"]))
        | RunnablePassthrough.assign(
            reranked_docs=lambda x: re_ranker.compress_documents(
                query=x["standalone_question"], documents=x["retrieved_docs"]
            ) if re_ranker else x["retrieved_docs"],
        )
        | {
            "answer": (
                RunnablePassthrough.assign(context=lambda x: format_for_final_prompt(x["reranked_docs"]))
                | ChatPromptTemplate.from_messages([("human", [*("{context}"), {"type": "text", "text": "\nQuestion: {question}"}])])
                | final_rag_llm_local
                | StrOutputParser()
            ),
            "image_paths": lambda x: [doc.page_content for doc in x["reranked_docs"] if doc.metadata.get('is_image', False)],
            "source_docs": lambda x: [doc for doc in x["reranked_docs"] if not doc.metadata.get('is_image', False)],
        }
    )

    print("\n🚀 CogniVerse is ready! Ask your multimodal questions.")
    print("-" * 50)
    
    chat_history = []
    while True:
        try:
            user_query = input("\nAsk a question: ")
            if user_query.lower() == 'exit':
                print("Goodbye! Happy studying.")
                break
            
            result = chain.invoke({"question": user_query, "chat_history": chat_history})
            
            chat_history.extend([
                HumanMessage(content=user_query),
                AIMessage(content=result["answer"]),
            ])

            print("\n--- Answer ---")
            print(result["answer"])
            
            if result["image_paths"]:
                print("\n--- Relevant Images Found ---")
                for path in result["image_paths"]:
                    print(f"- {Path(path).name} (Path: {path})")
                print("-----------------------------")

            print("\n--- Retrieved Text Sources ---")
            print(format_docs_for_display(result["source_docs"], []))
            print("----------------------------")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()

# LIMITAION: ADD BATCH IMAGE PROCESSING FOR LOCAL LLM
# # In src/cogniverse_app.py

# import os
# import uuid
# import base64
# import pickle
# from pathlib import Path
# from dotenv import load_dotenv
# import time

# # --- Core LangChain Imports (Unchanged) ---
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # --- Specific LangChain Component Imports (Unchanged) ---
# from langchain.storage import InMemoryStore
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain_cohere import CohereRerank

# from tqdm import tqdm

# # --- 1. CONFIGURATION and LOADING ---
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
# VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"
# IMAGE_DIR = PROCESSED_DATA_DIR / "images"

# # We are now back to using only local models for all required tasks.
# IMAGE_SUMMARY_MODEL_LOCAL = "llava"
# FINAL_RESPONSE_MODEL_LOCAL = "llava"
# QUESTION_CONDENSING_MODEL_LOCAL = "phi3:mini"

# # --- 2. HELPER FUNCTIONS ---
# def image_to_base64(image_path):
#     """Converts an image file to a base64 string."""
#     try:
#         with open(image_path, "rb") as img_file:
#             return base64.b64encode(img_file.read()).decode('utf-8')
#     except Exception as e:
#         print(f"Error encoding image {image_path}: {e}")
#         return None

# def generate_image_summary(llm, image_base64):
#     """Generates a text summary for a base64-encoded image using a local multimodal LLM."""
#     msg = llm.invoke(
#         [
#             HumanMessage(
#                 content=[
#                     {"type": "text", "text": "Provide a concise, one-sentence summary of this image's content. This summary will be used as a searchable index."},
#                     {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
#                 ]
#             )
#         ]
#     )
#     return msg.content

# def format_docs_for_display(docs, image_paths):
#     """A utility function to pretty-print the retrieved sources for the user."""
#     formatted_string = ""
#     text_docs = [doc for doc in docs if not doc.metadata.get('is_image', False)]
#     for i, doc in enumerate(text_docs):
#         source_page = doc.metadata.get('source_page', 'N/A')
#         formatted_string += f"\n--- Retrieved Text/Table {i+1} (Source: Page {source_page}) ---\n"
#         formatted_string += doc.page_content[:200] + "...\n"
#     if image_paths:
#         formatted_string += "\n--- Retrieved Images ---"
#         for path in image_paths:
#             formatted_string += f"\n- {Path(path).name}"
#     return formatted_string

# # --- 3. MAIN APPLICATION LOGIC ---

# def main():
#     print("--- Initializing CogniVerse Multimodal RAG Application ---")
#     # --- Step A: Load Pre-processed Data (WITH THE FIX) ---
#     try:
#         with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f:
#             texts = pickle.load(f)
#         with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f:
#             tables = pickle.load(f)

#         # *** THE FIX IS HERE: We now search for multiple common image extensions. ***
#         image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
#         image_paths = []
#         for ext in image_extensions:
#             image_paths.extend([str(p) for p in IMAGE_DIR.glob(ext)])
#         image_paths = sorted(list(set(image_paths))) # Sort and remove duplicates
#         print(f"Found {len(image_paths)} images in the directory.")
        
#     except FileNotFoundError:
#         print("\n❌ Pre-processed data not found. Please run `python src/data_processor.py` first.")
#         return

#     # --- Step B: Initialize LLMs (Local Only) ---
#     image_llm_local = OllamaLLM(model=IMAGE_SUMMARY_MODEL_LOCAL, temperature=0)
#     final_rag_llm_local = OllamaLLM(model=FINAL_RESPONSE_MODEL_LOCAL, temperature=0.1)
#     condense_llm_local = OllamaLLM(model=QUESTION_CONDENSING_MODEL_LOCAL, temperature=0)

#     # --- Step C: Setup the Multi-Vector Retriever (Unchanged) ---
#     print("Setting up the Multi-Vector Retriever...")
#     vectorstore = Chroma(
#         collection_name="cogniverse_hybrid_final_v3", # New collection name for a clean build
#         embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
#         persist_directory=str(VECTOR_STORE_PATH)
#     )
#     store = InMemoryStore()
#     id_key = "doc_id"
#     retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

#     # --- Step D: Populate the Retriever (WITH FAST HYBRID STRATEGY) ---
#     if not vectorstore.get()['ids']:
#         print("Vector store is empty. Populating with hybrid strategy (instant text, slow images)...")
        
#         # --- NEW HYBRID POPULATION LOGIC ---
#         # This is now incredibly fast for text and tables because it requires ZERO LLM calls.
        
#         # Combine texts and tables into one list of full documents
#         all_docs = [Document(page_content=t['text'], metadata={'source_page': t['source_page']}) for t in texts]
#         all_docs.extend([Document(page_content=t['html'], metadata={'source_page': t['source_page']}) for t in tables])
        
#         doc_ids = [str(uuid.uuid4()) for _ in all_docs]

#         # Store the FULL original documents in the docstore
#         retriever.docstore.mset(list(zip(doc_ids, all_docs)))

#         # Create the small "sub-chunks" to be vectorized. This requires NO LLM CALLS.
#         # We take the first 1024 characters of each large chunk. This is our "Table of Contents entry".
#         sub_chunk_docs = []
#         for i, doc in enumerate(tqdm(all_docs, desc="Creating Text Sub-Chunks")):
#             sub_content = doc.page_content[:1024]
#             sub_chunk_docs.append(Document(page_content=sub_content, metadata={id_key: doc_ids[i]}))
            
#         # Embed and store ONLY the small sub-chunks
#         BATCH_SIZE = 4000 # A safe number well below the 5461 limit.
#         for i in tqdm(range(0, len(sub_chunk_docs), BATCH_SIZE), desc="Adding Text Sub-Chunks to ChromaDB"):
#             batch = sub_chunk_docs[i:i+BATCH_SIZE]
#             retriever.vectorstore.add_documents(batch)
#         print("✅ Text and table vector store populated almost instantly.")

#         # --- Image Summarization (The only slow part, done locally) ---
#         print("Now, generating summaries for images using local LLaVA. This will take some time.")
#         valid_image_paths = [p for p in image_paths if image_to_base64(p) is not None]
#         image_base64s = [image_to_base64(p) for p in valid_image_paths]
        
#         image_summaries = []
#         for b64 in tqdm(image_base64s, desc="Summarizing Images"):
#             image_summaries.append(generate_image_summary(image_llm_local, b64))
#             # We add a small sleep here to give your system's resources a break between heavy LLM calls, preventing crashes.
#             time.sleep(1)

#         image_ids = [str(uuid.uuid4()) for _ in valid_image_paths]

#         # Store original image paths in docstore
#         original_images = [Document(page_content=p, metadata={'is_image': True}) for p in valid_image_paths]
#         retriever.docstore.mset(list(zip(image_ids, original_images)))
        
#         # Store image summaries in vectorstore
#         summary_docs = [Document(page_content=summary, metadata={id_key: image_ids[i]}) for i, summary in enumerate(image_summaries)]
#         retriever.vectorstore.add_documents(summary_docs)
        
#         vectorstore.persist()
#         print("✅ Retriever fully populated and vector store persisted.")
#     else:
#         print("✅ Vector store already populated. Loading from disk.")

#     # --- Step E: Setup the Re-ranker ---
#     re_ranker = None
#     if COHERE_API_KEY and "YOUR_TRIAL_API_KEY" not in COHERE_API_KEY:
#         print("✅ Cohere re-ranker is enabled.")
#         re_ranker = CohereRerank(
#             cohere_api_key=COHERE_API_KEY, top_n=4, model="rerank-english-v3.0"
#         )
#     else:
#         print("⚠️ Cohere re-ranker is disabled (API key not provided).")

#     # --- Step F: Define the Final Conversational RAG Chain ---
#     condense_question_prompt = ChatPromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:""")
    
#     condense_question_chain = condense_question_prompt | condense_llm_local | StrOutputParser()

#     def condense_question(input: dict):
#         if input.get("chat_history"):
#             return condense_question_chain
#         else:
#             return input["question"]

#     def format_for_final_prompt(docs):
#         prompt_content = []
#         prompt_content.append({"type": "text", "text": """You are an expert university study buddy. Your primary directive is to act as a tutor and give a justifiable, well-written, and easy-to-read answer based STRICTLY AND ONLY on the provided context, which may include text, tables, and images.

# **Instructions:**
# 1.  **Synthesize, Do Not Just List:** Read all the provided context documents. Weave the information into a **single, cohesive, flowing answer.** Do not list information from different documents separately. Your answer should read like a single, well-written explanation from an expert tutor.
# 2.  **Analyze and Explain Images/Tables:** If images or tables are present in the context, do not just mention them. **Analyze their content** and explain what they illustrate in relation to the user's question.
# 3.  **Format for Readability:** Use Markdown for formatting. Use headings and subheadings. **Bold** key terms and definitions. Use bullet points for lists.
# 4.  **Strictly Adhere to Context:** If the context does not contain enough information to answer the question, you MUST respond with exactly this phrase: "Based on the provided textbook, I cannot answer this question." and not a word more. Do not use any outside knowledge.

# --- CONTEXT START ---"""})

#         for doc in docs:
#             if doc.metadata.get('is_image', False):
#                 image_base64 = image_to_base64(doc.page_content)
#                 if image_base64:
#                     prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"})
#             else:
#                 source_page = doc.metadata.get('source_page', 'N/A')
#                 prompt_content.append({"type": "text", "text": f"\n[Text/Table from Page {source_page}]:\n{doc.page_content}"})

#         prompt_content.append({"type": "text", "text": "\n--- CONTEXT END ---\n"})
#         return prompt_content
    
#     chain = (
#         RunnablePassthrough.assign(standalone_question=condense_question)
#         | RunnablePassthrough.assign(retrieved_docs=lambda x: retriever.get_relevant_documents(x["standalone_question"]))
#         | RunnablePassthrough.assign(
#             reranked_docs=lambda x: re_ranker.compress_documents(
#                 query=x["standalone_question"], documents=x["retrieved_docs"]
#             ) if re_ranker else x["retrieved_docs"],
#         )
#         | {
#             "answer": (
#                 RunnablePassthrough.assign(context=lambda x: format_for_final_prompt(x["reranked_docs"]))
#                 | ChatPromptTemplate.from_messages([("human", [*("{context}"), {"type": "text", "text": "\nQuestion: {question}"}])])
#                 | final_rag_llm_local
#                 | StrOutputParser()
#             ),
#             "image_paths": lambda x: [doc.page_content for doc in x["reranked_docs"] if doc.metadata.get('is_image', False)],
#             "source_docs": lambda x: [doc for doc in x["reranked_docs"] if not doc.metadata.get('is_image', False)],
#         }
#     )

#     print("\n🚀 CogniVerse is ready! Ask your multimodal questions.")
#     print("-" * 50)
    
#     chat_history = []
#     while True:
#         try:
#             user_query = input("\nAsk a question: ")
#             if user_query.lower() == 'exit':
#                 print("Goodbye! Happy studying.")
#                 break
            
#             result = chain.invoke({"question": user_query, "chat_history": chat_history})
            
#             chat_history.extend([
#                 HumanMessage(content=user_query),
#                 AIMessage(content=result["answer"]),
#             ])

#             print("\n--- Answer ---")
#             print(result["answer"])
            
#             if result["image_paths"]:
#                 print("\n--- Relevant Images Found ---")
#                 for path in result["image_paths"]:
#                     print(f"- {Path(path).name} (Path: {path})")
#                 print("-----------------------------")

#             print("\n--- Retrieved Text Sources ---")
#             print(format_docs_for_display(result["source_docs"], []))
#             print("----------------------------")

#         except KeyboardInterrupt:
#             print("\n\nGoodbye!")
#             break
#         except Exception as e:
#             print(f"\n❌ An error occurred: {e}")

# if __name__ == "__main__":
#     main()
    
    
# LIMITATION : API RATE LIMITING
# import os
# import uuid
# import base64
# import pickle
# from pathlib import Path
# from dotenv import load_dotenv
# import time

# # --- Core LangChain Imports ---
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # --- Specific LangChain Component Imports ---
# from langchain.storage import InMemoryStore
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain_cohere import CohereRerank
# from langchain_google_genai import ChatGoogleGenerativeAI

# from tqdm import tqdm

# # --- 1. CONFIGURATION and LOADING ---
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
# VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"
# IMAGE_DIR = PROCESSED_DATA_DIR / "images"

# SUMMARY_MODEL_API = "gemini-1.5-flash"
# FINAL_RESPONSE_MODEL_LOCAL = "llava"
# QUESTION_CONDENSING_MODEL_LOCAL = "phi3:mini"

# # --- 2. HELPER FUNCTIONS ---

# def image_to_base64(image_path):
#     """Converts an image file to a base64 string."""
#     try:
#         with open(image_path, "rb") as img_file:
#             return base64.b64encode(img_file.read()).decode('utf-8')
#     except Exception as e:
#         print(f"Error encoding image {image_path}: {e}")
#         return None

# def generate_summary(llm, content, prompt_template):
#     """Generates a text summary for text or table content."""
#     prompt = ChatPromptTemplate.from_template(prompt_template)
#     chain = prompt | llm | StrOutputParser()
#     return chain.invoke({"element": content})

# def generate_image_summary(llm, image_base64):
#     """Generates a text summary for a base64-encoded image."""
#     msg = llm.invoke(
#         [
#             HumanMessage(
#                 content=[
#                     {"type": "text", "text": "Provide a concise, one-sentence summary of this image's content. This summary will be used as a searchable index."},
#                     {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
#                 ]
#             )
#         ]
#     )
#     return msg.content

# def format_docs_for_display(docs, image_paths):
#     """A utility function to pretty-print the retrieved sources for the user."""
#     formatted_string = ""
#     text_docs = [doc for doc in docs if not doc.metadata.get('is_image', False)]
#     for i, doc in enumerate(text_docs):
#         source_page = doc.metadata.get('source_page', 'N/A')
#         formatted_string += f"\n--- Retrieved Text/Table {i+1} (Source: Page {source_page}) ---\n"
#         formatted_string += doc.page_content[:200] + "...\n"
    
#     if image_paths:
#         formatted_string += "\n--- Retrieved Images ---"
#         for path in image_paths:
#             formatted_string += f"\n- {Path(path).name}"
#     return formatted_string

# # --- 3. MAIN APPLICATION LOGIC ---

# def main():
#     print("--- Initializing CogniVerse Multimodal RAG Application ---")

#     try:
#         with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f:
#             texts = pickle.load(f)
#         with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f:
#             tables = pickle.load(f)
#         image_paths = sorted([str(p) for p in IMAGE_DIR.glob("*.jpg")])
#     except FileNotFoundError:
#         print("\n❌ Pre-processed data not found. Please run `python src/data_processor.py` first.")
#         return

#     # --- Step B: Initialize LLMs ---
#     if not GOOGLE_API_KEY or "YOUR_NEW_GOOGLE_API_KEY" in GOOGLE_API_KEY:
#         print("❌ Error: GOOGLE_API_KEY not found or is a placeholder in .env file.")
#         return
        
#     summary_llm_api = ChatGoogleGenerativeAI(model=SUMMARY_MODEL_API, google_api_key=GOOGLE_API_KEY, temperature=0)
    
#     final_rag_llm_local = OllamaLLM(model=FINAL_RESPONSE_MODEL_LOCAL, temperature=0.1)
#     condense_llm_local = OllamaLLM(model=QUESTION_CONDENSING_MODEL_LOCAL, temperature=0)

#     # --- Step C: Setup the Multi-Vector Retriever ---
#     print("Setting up the Multi-Vector Retriever...")
#     vectorstore = Chroma(
#         collection_name="cogniverse_summaries_final",
#         embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
#         persist_directory=str(VECTOR_STORE_PATH)
#     )
#     store = InMemoryStore()
#     id_key = "doc_id"
#     retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

#     # --- Step D: Populate the Retriever (Fast API Version) ---
#     if not vectorstore.get()['ids']:
#         print("Vector store is empty. Populating with summaries and linking to original data...")
#         print("Generating summaries using Gemini API for speed. This should only take a few minutes.")
        
#         text_prompt = "Provide a concise, one-sentence summary of the following text that captures its main keywords and concepts. This summary will be used for a search index. Element: {element}"
#         table_prompt = "Provide a concise, one-sentence summary of the following table content that captures its main keywords and concepts. This summary will be used for a search index. Element: {element}"
        
#         text_summaries = []
#         for t in tqdm(texts, desc="Summarizing Texts with Gemini"):
#             text_summaries.append(generate_summary(summary_llm_api, t['text'], text_prompt))
#             time.sleep(5) # *** THE FIX: Wait for 5 seconds after each API call ***

#         table_summaries = []
#         for t in tqdm(tables, desc="Summarizing Tables with Gemini"):
#             table_summaries.append(generate_summary(summary_llm_api, t['html'], table_prompt))
#             time.sleep(5) # *** THE FIX: Wait for 5 seconds after each API call ***
        
#         valid_image_paths = [p for p in image_paths if image_to_base64(p) is not None]
#         image_base64s = [image_to_base64(p) for p in valid_image_paths]
        
#         image_summaries = []
#         for b64 in tqdm(image_base64s, desc="Summarizing Images with Gemini"):
#             image_summaries.append(generate_image_summary(summary_llm_api, b64))
#             time.sleep(5) # *** THE FIX: Wait for 5 seconds after each API call ***

#         text_ids = [str(uuid.uuid4()) for _ in texts]
#         table_ids = [str(uuid.uuid4()) for _ in tables]
#         image_ids = [str(uuid.uuid4()) for _ in valid_image_paths]
        
#         original_texts = [Document(page_content=t['text'], metadata={'source_page': t['source_page']}) for t in texts]
#         retriever.docstore.mset(list(zip(text_ids, original_texts)))
        
#         original_tables = [Document(page_content=t['html'], metadata={'source_page': t['source_page']}) for t in tables]
#         retriever.docstore.mset(list(zip(table_ids, original_tables)))
        
#         original_images = [Document(page_content=p, metadata={'is_image': True}) for p in valid_image_paths]
#         retriever.docstore.mset(list(zip(image_ids, original_images)))

#         summary_docs = []
#         for i, summary in enumerate(text_summaries):
#             summary_docs.append(Document(page_content=summary, metadata={id_key: text_ids[i]}))
#         for i, summary in enumerate(table_summaries):
#             summary_docs.append(Document(page_content=summary, metadata={id_key: table_ids[i]}))
#         for i, summary in enumerate(image_summaries):
#             summary_docs.append(Document(page_content=summary, metadata={id_key: image_ids[i]}))
        
#         retriever.vectorstore.add_documents(summary_docs)
#         vectorstore.persist()
#         print("✅ Retriever populated and vector store persisted.")
#     else:
#         print("✅ Vector store already populated. Loading from disk.")

#     # --- Step E: Setup the Re-ranker (Unchanged) ---
#     re_ranker = None
#     if COHERE_API_KEY and "YOUR_TRIAL_API_KEY" not in COHERE_API_KEY:
#         print("✅ Cohere re-ranker is enabled.")
#         re_ranker = CohereRerank(
#             cohere_api_key=COHERE_API_KEY, top_n=4, model="rerank-english-v3.0"
#         )
#     else:
#         print("⚠️ Cohere re-ranker is disabled (API key not provided).")

#     # --- Step F: Define the Final Conversational RAG Chain ---
#     condense_question_prompt = ChatPromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:""")
    
#     # *** BUG FIX 1: Use the correct variable name ***
#     condense_question_chain = condense_question_prompt | condense_llm_local | StrOutputParser()

#     def condense_question(input: dict):
#         if input.get("chat_history"):
#             return condense_question_chain
#         else:
#             return input["question"]

#     def format_for_final_prompt(docs):
#         prompt_content = []
#         prompt_content.append({"type": "text", "text": """You are an expert university study buddy. Your primary directive is to act as a tutor and give a justifiable, well-written, and easy-to-read answer based STRICTLY AND ONLY on the provided context, which may include text, tables, and images.

# **Instructions:**
# 1.  **Synthesize, Do Not Just List:** Read all the provided context documents. Weave the information into a **single, cohesive, flowing answer.** Do not list information from different documents separately. Your answer should read like a single, well-written explanation from an expert tutor.
# 2.  **Analyze and Explain Images/Tables:** If images or tables are present in the context, do not just mention them. **Analyze their content** and explain what they illustrate in relation to the user's question.
# 3.  **Format for Readability:** Use Markdown for formatting. Use headings and subheadings. **Bold** key terms and definitions. Use bullet points for lists.
# 4.  **Strictly Adhere to Context:** If the context does not contain enough information to answer the question, you MUST respond with exactly this phrase: "Based on the provided textbook, I cannot answer this question." and not a word more. Do not use any outside knowledge.

# --- CONTEXT START ---"""}) 
       
#         for doc in docs:
#             if doc.metadata.get('is_image', False):
#                 image_base64 = image_to_base64(doc.page_content)
#                 if image_base64:
#                     prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"})
#             else:
#                 source_page = doc.metadata.get('source_page', 'N/A')
#                 prompt_content.append({"type": "text", "text": f"\n[Text/Table from Page {source_page}]:\n{doc.page_content}"})
#         prompt_content.append({"type": "text", "text": "\n--- CONTEXT END ---\n"})
#         return prompt_content
    
#     chain = (
#         RunnablePassthrough.assign(standalone_question=condense_question)
#         | RunnablePassthrough.assign(retrieved_docs=lambda x: retriever.get_relevant_documents(x["standalone_question"]))
#         | RunnablePassthrough.assign(
#             reranked_docs=lambda x: re_ranker.compress_documents(
#                 query=x["standalone_question"], documents=x["retrieved_docs"]
#             ) if re_ranker else x["retrieved_docs"],
#         )
#         | {
#             "answer": (
#                 RunnablePassthrough.assign(context=lambda x: format_for_final_prompt(x["reranked_docs"]))
#                 | ChatPromptTemplate.from_messages([("human", [*("{context}"), {"type": "text", "text": "\nQuestion: {question}"}])])
#                 # *** BUG FIX 2: Use the correct variable name ***
#                 | final_rag_llm_local
#                 | StrOutputParser()
#             ),
#             "image_paths": lambda x: [doc.page_content for doc in x["reranked_docs"] if doc.metadata.get('is_image', False)],
#             "source_docs": lambda x: [doc for doc in x["reranked_docs"] if not doc.metadata.get('is_image', False)],
#         }
#     )

#     print("\n🚀 CogniVerse is ready! Ask your multimodal questions.")
#     print("-" * 50)
    
#     chat_history = []
#     while True:
#         try:
#             user_query = input("\nAsk a question: ")
#             if user_query.lower() == 'exit':
#                 print("Goodbye! Happy studying.")
#                 break
            
#             result = chain.invoke({"question": user_query, "chat_history": chat_history})
            
#             chat_history.extend([
#                 HumanMessage(content=user_query),
#                 AIMessage(content=result["answer"]),
#             ])

#             print("\n--- Answer ---")
#             print(result["answer"])
            
#             if result["image_paths"]:
#                 print("\n--- Relevant Images Found ---")
#                 for path in result["image_paths"]:
#                     print(f"- {Path(path).name} (Path: {path})")
#                 print("-----------------------------")

#             print("\n--- Retrieved Text Sources ---")
#             print(format_docs_for_display(result["source_docs"], []))
#             print("----------------------------")

#         except KeyboardInterrupt:
#             print("\n\nGoodbye!")
#             break
#         except Exception as e:
#             print(f"\n❌ An error occurred: {e}")

# if __name__ == "__main__":
#     main()