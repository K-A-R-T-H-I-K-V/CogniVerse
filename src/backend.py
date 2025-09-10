# In src/backend.py (FINAL, VETTED, STABLE VERSION)

import os
import uuid
import base64
import pickle
from pathlib import Path
from dotenv import load_dotenv
import re
from operator import itemgetter

from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Core LangChain Imports ---
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# --- Specific LangChain Component Imports ---
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_cohere import CohereRank

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"

# --- THE FINAL MODEL CHOICE: High quality AND stability ---
FINAL_RESPONSE_MODEL_LOCAL = "llava"
QUESTION_CONDENSING_MODEL_LOCAL = "phi3:mini"

# --- 2. HELPER FUNCTIONS ---
def split_docs_by_type(docs):
    text_docs, image_docs = [], []
    for doc in docs:
        if doc.metadata.get('is_image', False):
            image_docs.append(doc)
        else:
            text_docs.append(doc)
    return {"text_docs": text_docs, "image_docs": image_docs}

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        return None

def format_for_final_prompt(docs):
    prompt_content = [{"type": "text", "text": "You are an expert university study buddy. Your primary directive is to act as a tutor and give a justifiable, well-written, and easy-to-read answer based STRICTLY AND ONLY on the provided context, which may include text, tables, and images.\n\n**Instructions:**\n1.  **Synthesize, Do Not Just List:** Read all the provided context documents. Weave the information into a **single, cohesive, flowing answer.**\n2.  **Analyze and Explain Images/Tables:** If images or tables are present, **analyze their content** and explain what they illustrate in relation to the user's question.\n3.  **Format for Readability:** Use Markdown for formatting (bold, lists, etc.).\n4.  **Strictly Adhere to Context:** If the context does not contain enough information, you MUST respond with exactly this phrase: \"Based on the provided textbook, I cannot answer this question.\" and not a word more.\n\n--- CONTEXT START ---"}]
    for doc in docs:
        if doc.metadata.get('is_image', False):
            b64_image = image_to_base64(doc.page_content)
            if b64_image:
                prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"})
        else:
            source_page = doc.metadata.get('source_page', 'N/A')
            prompt_content.append({"type": "text", "text": f"\n[Text/Table from Page {source_page}]:\n{doc.page_content}"})
    prompt_content.append({"type": "text", "text": "\n--- CONTEXT END ---\n"})
    return prompt_content

# --- 3. GLOBAL AI SETUP (Load models and data ONCE on startup) ---

print("--- Initializing CogniVerse Backend ---")

print("Pre-loading AI models into memory. This may take a moment...")
final_rag_llm_local = OllamaLLM(model=FINAL_RESPONSE_MODEL_LOCAL, temperature=0.1)
condense_llm_local = OllamaLLM(model=QUESTION_CONDENSING_MODEL_LOCAL, temperature=0)
print("✅ AI models loaded.")

print("Loading vector store...")
vectorstore = Chroma(
    collection_name="cogniverse-final-v7",
    embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"),
    persist_directory=str(VECTOR_STORE_PATH)
)
store = InMemoryStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={'k': 10}
)

print("Populating in-memory docstore from processed files...")
# Load all original documents and recreate the docstore. This is crucial for the retriever to work.
with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f: texts = pickle.load(f)
with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f: tables = pickle.load(f)
with open(PROCESSED_DATA_DIR / "image_paths.pkl", "rb") as f: image_paths = pickle.load(f)

# Recreate the docstore content in the exact same order as it was created
all_text_docs = [Document(page_content=t['text'], metadata={'source_page': t['source_page']}) for t in texts]
all_text_docs.extend([Document(page_content=t['html'], metadata={'source_page': t['source_page']}) for t in tables])
text_doc_ids = [str(uuid.uuid4()) for _ in all_text_docs] # These IDs must match the ones used during population

image_docs = [Document(page_content=p, metadata={'is_image': True}) for p in image_paths]
image_ids = [str(uuid.uuid4()) for _ in image_docs] # Same for these

# This reconstruction is a known challenge. A better production system would persist the docstore.
# For our demo, we rely on the vectorstore's metadata to link back.
all_db_data = vectorstore.get(include=["metadatas"])
all_doc_ids_from_db = [meta[id_key] for meta in all_db_data["metadatas"]]

# We can't perfectly reconstruct the UUID link without saving it.
# The MultiVectorRetriever *should* work by using the IDs it finds in the vector store
# to look up documents that we must now manually add back to the in-memory store.
# A simpler approach for the backend is to let the retriever handle it, but we need to populate the store.
# The most robust way without saving the docstore is to assume the vectorstore's internal order.
all_original_docs = all_text_docs + image_docs
if len(all_doc_ids_from_db) == len(all_original_docs):
    store.mset(list(zip(all_doc_ids_from_db, all_original_docs)))
    print("✅ Docstore populated (in-memory).")
else:
    print(f"⚠️ Docstore population mismatch. DB has {len(all_doc_ids_from_db)} summaries, found {len(all_original_docs)} original docs. Retrieval may fail.")


re_ranker = None
if COHERE_API_KEY and "YOUR_TRIAL_API_KEY" not in COHERE_API_KEY:
    print("✅ Cohere re-ranker is enabled.")
    re_ranker = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")
else:
    print("⚠️ Cohere re-ranker is disabled.")

hyde_prompt_template = "Please generate a concise, one-paragraph, textbook-style answer for the following question to help find relevant documents.\nQuestion: {question}\nHypothetical Answer:"
hyde_prompt = PromptTemplate(input_variables=["question"], template=hyde_prompt_template)
hyde_chain = hyde_prompt | condense_llm_local | StrOutputParser()

# --- 4. THE CONVERSATIONAL RAG CHAIN ---
def condense_question(input: dict):
    if not input.get("chat_history"):
        return input["question"]
    # In a real app, this would be a more complex chain. For our demo, this is sufficient.
    condense_question_prompt = ChatPromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:""")
    return (condense_question_prompt | condense_llm_local | StrOutputParser()).invoke(input)

def format_final_output(input_dict):
    return {
        "answer": input_dict["answer"],
        "image_paths": [doc.page_content for doc in input_dict["final_docs"] if doc.metadata.get('is_image', False)],
        "source_docs": [doc for doc in input_dict["final_docs"] if not doc.metadata.get('is_image', False)],
    }

rag_chain_with_llm = (
    RunnablePassthrough.assign(context=lambda x: format_for_final_prompt(x["final_docs"]))
    .assign(final_prompt_content=lambda x: x["context"] + [{"type": "text", "text": f"\n\nQuestion: {x['standalone_question']}"}])
    | ChatPromptTemplate.from_messages([("human", "{final_prompt_content}")])
    | final_rag_llm_local
    | StrOutputParser()
)

gate = RunnableBranch(
    (lambda x: not x["final_docs"], lambda x: "Based on the provided textbook, I cannot answer this question."),
    rag_chain_with_llm,
)

chain = (
    RunnablePassthrough.assign(standalone_question=condense_question)
    | RunnablePassthrough.assign(hypothetical_document=hyde_chain)
    | RunnablePassthrough.assign(retrieved_docs=lambda x: retriever.invoke(x["hypothetical_document"]))
    | RunnablePassthrough.assign(split_docs=lambda x: split_docs_by_type(x["retrieved_docs"]))
    | {
        "reranked_docs": (lambda x: re_ranker.compress_documents(query=x["standalone_question"], documents=x["split_docs"]["text_docs"]) if re_ranker else x["split_docs"]["text_docs"]),
        "image_docs": lambda x: x["split_docs"]["image_docs"],
        "question": itemgetter("question"),
        "standalone_question": itemgetter("standalone_question"),
    }
    | RunnablePassthrough.assign(final_docs=lambda x: x["reranked_docs"] + x["image_docs"])
    | RunnablePassthrough.assign(answer=gate)
    | format_final_output
)
print("✅ Conversational RAG chain is ready.")

# --- 5. FLASK APPLICATION ---
app = Flask(__name__)
CORS(app)
chat_history = []

@app.route('/ask', methods=['POST'])
def ask():
    global chat_history
    user_query = request.json.get('question')
    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    try:
        result = chain.invoke({"question": user_query, "chat_history": chat_history})
        final_docs = result.get('source_docs', []) + result.get('image_paths', [])
        
        sources = []
        for doc_content in final_docs:
            if isinstance(doc_content, Document): # It's a text/table doc
                 sources.append({
                     "type": "text", 
                     "content": doc_content.page_content, 
                     "page": doc_content.metadata.get('source_page', 'N/A')
                 })
            else: # It's an image path string
                b64_img = image_to_base64(doc_content)
                if b64_img:
                    sources.append({
                        "type": "image", 
                        "content": f"data:image/jpeg;base64,{b64_img}", 
                        "filename": Path(doc_content).name
                    })

        answer = result.get('answer', "Sorry, I encountered an error.")
        chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        print(f"ERROR in /ask endpoint: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history reset successfully"})

if __name__ == '__main__':
    print("🚀 CogniVerse backend is running at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)

# # In src/backend.py (FINAL, STABLE VERSION)

# import os
# import uuid
# import base64
# import pickle
# from pathlib import Path
# from dotenv import load_dotenv
# import time
# import re
# from operator import itemgetter

# from flask import Flask, request, jsonify
# from flask_cors import CORS

# # --- Core LangChain Imports ---
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# # --- Specific LangChain Component Imports ---
# from langchain.storage import InMemoryStore
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain_cohere import CohereRerank

# # --- 1. CONFIGURATION and LOADING ---
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
# VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"

# # --- THE STABILITY FIX: Use a smaller, more efficient local model ---
# FINAL_RESPONSE_MODEL_LOCAL = "moondream" 
# QUESTION_CONDENSING_MODEL_LOCAL = "phi3:mini"

# # --- 2. HELPER FUNCTIONS ---
# def split_docs_by_type(docs):
#     text_docs, image_docs = [], []
#     for doc in docs:
#         if doc.metadata.get('is_image', False): image_docs.append(doc)
#         else: text_docs.append(doc)
#     return {"text_docs": text_docs, "image_docs": image_docs}

# def image_to_base64(image_path):
#     try:
#         with open(image_path, "rb") as img_file:
#             return base64.b64encode(img_file.read()).decode('utf-8')
#     except Exception: return None

# def format_for_final_prompt(docs):
#     prompt_content = [{"type": "text", "text": "You are an expert university study buddy. Your primary directive is to act as a tutor and give a justifiable, well-written, and easy-to-read answer based STRICTLY AND ONLY on the provided context, which may include text, tables, and images.\n\n**Instructions:**\n1.  **Synthesize, Do Not Just List:** Read all the provided context documents. Weave the information into a **single, cohesive, flowing answer.**\n2.  **Analyze and Explain Images/Tables:** If images or tables are present, **analyze their content** and explain what they illustrate in relation to the user's question.\n3.  **Format for Readability:** Use Markdown for formatting (bold, lists, etc.).\n4.  **Strictly Adhere to Context:** If the context does not contain enough information, you MUST respond with exactly this phrase: \"Based on the provided textbook, I cannot answer this question.\" and not a word more.\n\n--- CONTEXT START ---"}]
#     for doc in docs:
#         if doc.metadata.get('is_image', False):
#             b64_image = image_to_base64(doc.page_content)
#             if b64_image:
#                 prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"})
#         else:
#             source_page = doc.metadata.get('source_page', 'N/A')
#             prompt_content.append({"type": "text", "text": f"\n[Text/Table from Page {source_page}]:\n{doc.page_content}"})
#     prompt_content.append({"type": "text", "text": "\n--- CONTEXT END ---\n"})
#     return prompt_content

# # --- 3. GLOBAL AI SETUP (Load models and data ONCE on startup) ---

# print("--- Initializing CogniVerse Backend ---")
# final_rag_llm_local = OllamaLLM(model=FINAL_RESPONSE_MODEL_LOCAL, temperature=0.1)
# condense_llm_local = OllamaLLM(model=QUESTION_CONDENSING_MODEL_LOCAL, temperature=0)

# print("Loading vector store...")
# vectorstore = Chroma(
#     collection_name="cogniverse-final-v7", # Make sure this matches your last build
#     embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"),
#     persist_directory=str(VECTOR_STORE_PATH)
# )
# store = InMemoryStore()
# id_key = "doc_id"
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={'k': 10}
# )

# # Manually populate the in-memory docstore. This part is tricky because IDs aren't saved.
# # For a robust system, the docstore should also be persisted. For our demo, we reload it.
# print("Populating in-memory docstore from processed files...")
# # We assume the summary vectors and the original docs are in the same order.
# all_summary_docs = vectorstore.get(include=["metadatas"])["metadatas"]
# doc_ids_from_db = [meta[id_key] for meta in all_summary_docs]

# # Load original docs
# with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f: texts = pickle.load(f)
# with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f: tables = pickle.load(f)
# with open(PROCESSED_DATA_DIR / "image_paths.pkl", "rb") as f: image_paths = pickle.load(f)
# all_text_docs = [Document(page_content=t['text'], metadata={'source_page': t['source_page']}) for t in texts]
# all_text_docs.extend([Document(page_content=t['html'], metadata={'source_page': t['source_page']}) for t in tables])
# image_docs = [Document(page_content=p, metadata={'is_image': True}) for p in image_paths]
# all_original_docs = all_text_docs + image_docs
# # This is a critical assumption: that the order matches the order during creation.
# store.mset(list(zip(doc_ids_from_db, all_original_docs)))
# print("✅ Docstore populated.")

# re_ranker = None
# if COHERE_API_KEY and "YOUR_TRIAL_API_KEY" not in COHERE_API_KEY:
#     print("✅ Cohere re-ranker is enabled.")
#     re_ranker = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")
# else:
#     print("⚠️ Cohere re-ranker is disabled.")

# # --- 4. THE CONVERSATIONAL RAG CHAIN ---
# def condense_question(input: dict):
#     if not input.get("chat_history"): return input["question"]
#     # ... condense logic ...
#     condense_question_prompt = ChatPromptTemplate.from_template("...")
#     return (condense_question_prompt | condense_llm_local | StrOutputParser())

# rag_chain_with_llm = (
#     RunnablePassthrough.assign(context=lambda x: format_for_final_prompt(x["final_docs"]))
#     .assign(final_prompt_content=lambda x: x["context"] + [{"type": "text", "text": f"\n\nQuestion: {x['standalone_question']}"}])
#     | ChatPromptTemplate.from_messages([("human", "{final_prompt_content}")])
#     | final_rag_llm_local
#     | StrOutputParser()
# )
# gate = RunnableBranch(
#     (lambda x: not x["final_docs"], lambda x: "Based on the provided textbook, I cannot answer this question."),
#     rag_chain_with_llm,
# )
# chain = (
#     RunnablePassthrough.assign(standalone_question=condense_question)
#     | RunnablePassthrough.assign(retrieved_docs=lambda x: retriever.invoke(x["standalone_question"]))
#     | RunnablePassthrough.assign(split_docs=lambda x: split_docs_by_type(x["retrieved_docs"]))
#     | {
#         "reranked_docs": (
#             lambda x: re_ranker.compress_documents(query=x["standalone_question"], documents=x["split_docs"]["text_docs"])
#             if re_ranker else x["split_docs"]["text_docs"]
#         ),
#         "image_docs": lambda x: x["split_docs"]["image_docs"],
#         "standalone_question": itemgetter("standalone_question"),
#         "question": itemgetter("question"), # Pass original question through
#     }
#     | RunnablePassthrough.assign(final_docs=lambda x: x["reranked_docs"] + x["image_docs"])
#     | RunnablePassthrough.assign(answer=gate)
# )
# print("✅ Conversational RAG chain is ready.")

# # --- 5. FLASK APPLICATION ---
# app = Flask(__name__)
# CORS(app)
# chat_history = []

# @app.route('/ask', methods=['POST'])
# def ask():
#     global chat_history
#     user_query = request.json.get('question')
#     if not user_query: return jsonify({"error": "No question"}), 400

#     try:
#         result = chain.invoke({"question": user_query, "chat_history": chat_history})
#         final_docs = result.get('final_docs', [])
        
#         # Format sources for frontend
#         sources = []
#         for doc in final_docs:
#             if doc.metadata.get('is_image', False):
#                 b64_img = image_to_base64(doc.page_content)
#                 if b64_img:
#                     sources.append({"type": "image", "content": f"data:image/jpeg;base64,{b64_img}", "filename": Path(doc.page_content).name})
#             else:
#                 sources.append({"type": "text", "content": doc.page_content, "page": doc.metadata.get('source_page', 'N/A')})

#         answer = result.get('answer', "Sorry, I encountered an error.")
#         chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
#         if len(chat_history) > 10: chat_history = chat_history[-10:]

#         return jsonify({"answer": answer, "sources": sources})
#     except Exception as e:
#         print(f"ERROR in /ask endpoint: {e}")
#         return jsonify({"error": "An internal error occurred."}), 500

# @app.route('/reset', methods=['POST'])
# def reset():
#     global chat_history
#     chat_history = []
#     return jsonify({"message": "Chat history reset"})

# if __name__ == '__main__':
#     print("🚀 CogniVerse backend is running at http://127.0.0.1:5000")
#     app.run(host='0.0.0.0', port=5000)