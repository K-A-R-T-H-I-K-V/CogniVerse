# In src/cogniverse_app.py

import os
import uuid
import base64
import pickle
from pathlib import Path
from dotenv import load_dotenv
import time
import re
# In src/cogniverse_app.py, add this with the other imports
from operator import itemgetter

# --- Core LangChain Imports ---
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Add this with the other LangChain imports
from langchain_core.runnables import RunnableBranch

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
# In src/cogniverse_app.py, add this with the other helper functions

def split_docs_by_type(docs):
    """Splits a list of documents into text/table docs and image docs."""
    text_docs, image_docs = [], []
    for doc in docs:
        if doc.metadata.get('is_image', False):
            image_docs.append(doc)
        else:
            text_docs.append(doc)
    return {"text_docs": text_docs, "image_docs": image_docs}

def image_to_base64(image_path):
    """Converts an image file to a base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# In src/cogniverse_app.py, ADD THIS NEW HELPER FUNCTION

def parse_llm_summaries(llm_output: str, expected_count: int) -> list[str]:
    """
    A robust parser for the LLM's numbered list output.
    It guarantees a list of the correct length and handles common LLM failures.
    """
    lines = llm_output.strip().split('\n')
    summary_map = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for lines that start with a number like "1." or "1: "
        match = re.match(r'^(\d+)[.:\s]\s*(.*)', line)
        if match:
            num = int(match.group(1))
            summary_text = match.group(2).strip()
            if 1 <= num <= expected_count:
                summary_map[num] = summary_text

    # Build the final list, ensuring 1-to-1 mapping and correct length.
    # Your brilliant "NaN" idea is implemented here.
    final_summaries = []
    for i in range(1, expected_count + 1):
        final_summaries.append(summary_map.get(i, "[SUMMARY_GENERATION_FAILED]"))
        
    return final_summaries

# In src/cogniverse_app.py, REPLACE the old function with this FINAL version.

def generate_image_summaries_in_batch(llm, image_batch_b64):
    """
    Generates summaries for a batch of images.
    This FINAL version uses a numbered prompt and a robust parser for maximum reliability.
    """
    if not image_batch_b64:
        return []

    # --- THE ULTIMATE PROMPT ---
    prompt_text = f"""
<instructions>
You are an expert at analyzing textbook figures. You will be given {len(image_batch_b64)} images.
Your task is to provide a concise, one-sentence summary for EACH image.
- You MUST number your summaries starting from 1.
- The summary on line N MUST correspond to IMAGE N.
- The summary MUST capture the main keywords and concepts shown in the image.
- Do NOT add any extra text, conversation, or apologies.
- If you cannot analyze an image, output the number and the exact text: [SUMMARY_GENERATION_FAILED]
</instructions>

<example>
IMAGE 1 is a diagram.
IMAGE 2 is a graph.
Your output MUST be in this exact format given below, and NOT A WORD MORE or LESS:
1. A diagram illustrating the client-server architecture in cloud computing.
2. A graph showing the performance scaling of parallel processing tasks.
</example>

Here are the {len(image_batch_b64)} images:
"""
    
    prompt_content = [{"type": "text", "text": prompt_text}]
    for b64 in image_batch_b64:
        prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"})

    try:
        msg = llm.invoke([HumanMessage(content=prompt_content)])
        
        # Use our new robust parser to handle the output
        summaries = parse_llm_summaries(msg.content, len(image_batch_b64))
        
        # Final check to prevent any downstream errors, though the parser should handle this.
        if len(summaries) != len(image_batch_b64):
             print(f"\nFATAL PARSER ERROR: Count mismatch after parsing. Expected {len(image_batch_b64)}, got {len(summaries)}. Returning placeholders.")
             return ["[SUMMARY_PARSER_ERROR]"] * len(image_batch_b64)
                
        return summaries
        
    except Exception as e:
        print(f"\nAn API error occurred during batch image summarization: {e}")
        return ["[SUMMARY_API_ERROR]"] * len(image_batch_b64)
        
    except Exception as e:
        print(f"\nAn error occurred during batch image summarization: {e}")
        # If the entire API call fails, return a list of placeholders of the correct length.
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
        # embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"),
        persist_directory=str(VECTOR_STORE_PATH)
    )
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, 
        docstore=store, 
        id_key=id_key,
        search_kwargs={'k': 10}
    )

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
            # sub_chunk_docs.append(Document(page_content=doc.page_content[:1024], metadata={id_key: doc_ids[i]}))
            sub_chunk_docs.append(Document(page_content=doc.page_content, metadata={id_key: doc_ids[i]}))
        
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

            time.sleep(5) 
        
        image_ids = [str(uuid.uuid4()) for _ in valid_image_paths]
        image_docs = [Document(page_content=p, metadata={'is_image': True}) for p in valid_image_paths]
        retriever.docstore.mset(list(zip(image_ids, image_docs)))
        
        summary_docs = [Document(page_content=summary, metadata={id_key: image_ids[i]}) for i, summary in enumerate(image_summaries)]
        retriever.vectorstore.add_documents(summary_docs)
        
        # REPLACE IT WITH THIS LINE:
        del vectorstore
        print("✅ Vector store populated and saved to disk.")
        print("✅ Retriever fully populated and vector store persisted.")
    else:
        print("✅ Vector store already populated. Loading from disk.")

    # --- Step E: Setup the Re-ranker ---
    re_ranker = None
    if COHERE_API_KEY and "YOUR_TRIAL_API_KEY" not in COHERE_API_KEY:
        print("✅ Cohere re-ranker is enabled.")
        re_ranker = CohereRerank(
            cohere_api_key=COHERE_API_KEY, 
            top_n=3, 
            model="rerank-english-v3.0"
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
    
    # In src/cogniverse_app.py, REPLACE the entire chain block with this one

    # In src/cogniverse_app.py, REPLACE the entire chain block with this one

    # This helper function will format our final output
    def format_final_output(input_dict):
        return {
            "answer": input_dict["answer"],
            "image_paths": [doc.page_content for doc in input_dict["final_docs"] if doc.metadata.get('is_image', False)],
            "source_docs": [doc for doc in input_dict["final_docs"] if not doc.metadata.get('is_image', False)],
        }

    # The main logic for handling a query when context IS found
    rag_chain_with_llm = (
        RunnablePassthrough.assign(context=lambda x: format_for_final_prompt(x["final_docs"]))
        .assign(
            final_prompt_content=lambda x: x["context"] + [{"type": "text", "text": f"\n\nQuestion: {x['question']}"}]
        )
        | ChatPromptTemplate.from_messages([
            ("human", "{final_prompt_content}")
        ])
        | final_rag_llm_local
        | StrOutputParser()
    )

    # The "gate" - a branch that checks if we found any documents
    gate = RunnableBranch(
        (lambda x: not x["final_docs"], lambda x: "Based on the provided textbook, I cannot answer this question."),
        rag_chain_with_llm,
    )

    # The full, final chain
    chain = (
        RunnablePassthrough.assign(standalone_question=condense_question)
        | RunnablePassthrough.assign(retrieved_docs=lambda x: retriever.invoke(x["standalone_question"]))
        | RunnablePassthrough.assign(split_docs=lambda x: split_docs_by_type(x["retrieved_docs"]))
        | {
            "reranked_docs": (
                lambda x: re_ranker.compress_documents(
                    query=x["standalone_question"], documents=x["split_docs"]["text_docs"]
                ) if re_ranker else x["split_docs"]["text_docs"]
            ),
            "image_docs": lambda x: x["split_docs"]["image_docs"],
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(final_docs=lambda x: x["reranked_docs"] + x["image_docs"])
        | RunnablePassthrough.assign(answer=gate) # Pass the results to the gate
        | format_final_output # Format the final dictionary for display
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
            
            print("--- DEBUG: RAW CHAIN OUTPUT ---")
            print(result)
            print("-----------------------------")

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

