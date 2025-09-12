# In src/backend.py (THE FINAL, WORKING, PERSISTENT VERSION)

import os
import uuid
import base64
import pickle
from pathlib import Path
from dotenv import load_dotenv
import re
from operator import itemgetter
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from tqdm import tqdm

# --- Core LangChain Imports ---
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# --- Specific LangChain Component Imports ---
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_cohere import CohereRerank
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"
DOC_STORE_PATH = PROJECT_ROOT / "doc_store" 

FINAL_RESPONSE_MODEL_LOCAL = "llava"
QUESTION_CONDENSING_MODEL_LOCAL = "phi3:mini"
SUMMARY_MODEL_API = "gemini-1.5-flash"
summary_llm_api = ChatGoogleGenerativeAI(model=SUMMARY_MODEL_API, google_api_key=GOOGLE_API_KEY, temperature=0)
final_rag_llm_api = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1) # Or "gemini-1.5-pro" for max quality

# --- 2. HELPER FUNCTIONS ---
def parse_llm_summaries(llm_output: str, expected_count: int) -> list[str]:
    lines = llm_output.strip().split('\n')
    summary_map = {}
    for line in lines:
        match = re.match(r'^(\d+)[.:\s]\s*(.*)', line.strip())
        if match:
            num, summary_text = int(match.group(1)), match.group(2).strip()
            if 1 <= num <= expected_count:
                summary_map[num] = summary_text
    return [summary_map.get(i, "[SUMMARY_GENERATION_FAILED]") for i in range(1, expected_count + 1)]

def generate_text_summaries_in_batch(llm, text_batch: list[str]) -> list[str]:
    if not text_batch: return []
    prompt_text = f"""<instructions>
You are an expert at summarizing academic text for a search index. You will be given a batch of {len(text_batch)} text chunks from a textbook.
Your task is to provide a concise, one-sentence summary for EACH text chunk.
- You MUST number your summaries starting from 1.
- The summary on line N MUST correspond to CHUNK N.
- The summary MUST capture the main keywords, definitions, and concepts in the chunk.
- Do NOT add any extra text, conversation, or apologies.
- If a chunk is nonsensical or too short to summarize, output the number and the exact text: [SUMMARY_GENERATION_FAILED]
</instructions>
<example>
--- CHUNK 1 ---
4.1.1 Public, Private, and Hybrid Clouds. The concept of cloud computing has evolved from cluster, grid, and utility computing.
--- CHUNK 2 ---
A public cloud is owned by the public and accessible by any user who has paid for the service.
Your output MUST be in this exact format:
1. An introduction to cloud computing, its evolution from cluster and grid computing, and the three main types: public, private, and hybrid.
2. A definition of a public cloud as a service owned by providers and accessible on a pay-per-use basis.
</example>
Here are the {len(text_batch)} text chunks:
"""
    for i, text in enumerate(text_batch):
        prompt_text += f"--- CHUNK {i+1} ---\n{text}\n\n"
    try:
        msg = llm.invoke(prompt_text)
        return parse_llm_summaries(msg.content, len(text_batch))
    except Exception as e:
        print(f"Error in text summarization: {e}")
        return ["[SUMMARY_API_ERROR]"] * len(text_batch)

def generate_image_summaries_in_batch(llm, image_batch_b64: list) -> list[str]:
    if not image_batch_b64: return []
    prompt_text = f"""<instructions>
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
        return parse_llm_summaries(msg.content, len(image_batch_b64))
    except Exception as e:
        print(f"Error in image summarization: {e}")
        return ["[SUMMARY_API_ERROR]"] * len(image_batch_b64)

def split_docs_by_type(docs):
    text_docs, image_docs = [], []
    for doc in docs:
        if doc.metadata.get('is_image', False): image_docs.append(doc)
        else: text_docs.append(doc)
    return {"text_docs": text_docs, "image_docs": image_docs}

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file: return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception: return None

# In src/backend.py

# def format_for_final_prompt(docs):
#     prompt_content = [{
#         "type": "text",
#         "text": """You are a world-class university study buddy AI. Your sole purpose is to answer a student's question based *exclusively* on the textbook excerpts provided to you.

# **--- PRIMARY DIRECTIVE ---**
# - You are strictly forbidden from using any external knowledge.
# - Your entire answer MUST be derived from the provided CONTEXT.
# - If the CONTEXT does not contain enough information to answer the question, you MUST reply with the single, exact phrase: "Based on the provided textbook, I cannot answer this question."
# - Do not apologize or add any conversational filler.
# - If images or tables are provided, analyze them and integrate their information into your answer. Refer to an image as "the provided diagram" or "the figure illustrates...".
# - If you discuss a diagram, insert the placeholder `[DIAGRAM]` exactly where the image should appear in your explanation.

# --- START OF CONTEXT ---"""
#     }]

#     has_image = False
#     for doc in docs:
#         if doc.metadata.get('is_image', False):
#             has_image = True
#             b64_image = image_to_base64(doc.page_content)
#             if b64_image:
#                 prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"})
#         else:
#             source_page = doc.metadata.get('source_page', 'N/A')
#             prompt_content.append({"type": "text", "text": f"\n\n[Text from Page {source_page}]:\n{doc.page_content}"})

#     # This is the new, crucial part. The task is defined AFTER all context.
#     final_instruction = """--- END OF CONTEXT ---

# **Final Task:** Based ONLY on the context provided above, synthesize a comprehensive and clear answer to the following question. Start your answer with "According to the textbook...".
# """
#     prompt_content.append({"type": "text", "text": final_instruction})
    
#     return prompt_content


# def format_for_final_prompt(docs):
#     # 1. Build the context string from text documents
#     context_str = ""
#     for doc in docs:
#         if not doc.metadata.get('is_image', False):
#             source_page = doc.metadata.get('source_page', 'N/A')
#             context_str += f"\n\n<source page=\"{source_page}\">\n{doc.page_content}\n</source>"

#     # 2. Find the first image for the multimodal part of the prompt
#     b64_image = None
#     first_image_doc = next((doc for doc in docs if doc.metadata.get('is_image')), None)
#     if first_image_doc:
#         b64_image = image_to_base64(first_image_doc.page_content)

#     # 3. Construct the final "Expert Educator" prompt
#     prompt_text = f"""<role>
# You are an Expert Educator AI. Your purpose is to create clear, structured, and easy-to-understand study notes for a university student based *strictly* on their textbook material.
# </role>

# <rules>
# 1.  Your entire response MUST be derived from the provided <context>, which includes both text and images. The image is a valid and critical source of information.
# 2.  **Formatting is crucial.** You MUST use Markdown for clear formatting. Use numbered or bulleted lists for components, steps, or key features. Use bold text for key terms.
# 3.  **Preserve Key Terminology.** When listing components, state the main term exactly as it appears in the textbook.
# 4.  **Explain Simply.** Immediately after stating the key term, provide a "Simple Explanation" in your own words to clarify the concept for the student.
# 5.  **Analyze Images.** If an image is provided, you MUST analyze it. Seamlessly integrate its description into the relevant part of your explanation and mark its location with the `[DIAGRAM]` placeholder.
# 6.  If the context is insufficient, reply ONLY with: "Based on the provided textbook, I cannot answer this question."
# </rules>

# <example_format>
# Here is an example of the desired output format:
# According to the textbook, the main components are:

# 1.  **Message:**
#     * **Simple Explanation:** This is the actual information being sent, like an email, a picture, or a video.
# 2.  **Sender:**
#     * **Simple Explanation:** This is the device or person that originates the message, such as your phone or computer.
#     * The provided diagram illustrates the sender on the left, initiating the communication process. `[DIAGRAM]`
# </example_format>

# <context>{context_str}
# </context>

# <instructions>
# Now, following all the rules and using the format shown in the example, create a clear and structured answer to the user's question based only on the context provided.
# </instructions>

# <question>
# """

#     prompt_content = [{"type": "text", "text": prompt_text}]

#     # 4. Add the image to the prompt content if it exists
#     if b64_image:
#         prompt_content.insert(1, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"})
#         prompt_content.insert(2, {"type": "text", "text": "\nAn image is provided in the context. You must analyze it as per the rules."})
    
#     return prompt_content

# In src/backend.py

# In src/backend.py

def prepare_final_prompt_content(docs):
    """Prepares the list of text and image parts for the final prompt."""
    context_str = ""
    for doc in docs:
        if not doc.metadata.get('is_image', False):
            source_page = doc.metadata.get('source_page', 'N/A')
            context_str += f"\n\n<source page=\"{source_page}\">\n{doc.page_content}\n</source>"

    b64_image = None
    first_image_doc = next((doc for doc in docs if doc.metadata.get('is_image')), None)
    if first_image_doc:
        b64_image = image_to_base64(first_image_doc.page_content)

    prompt_text = f"""<role>
You are an Expert Educator AI creating clear, structured study notes.
</role>
<rules>
1. Base your entire response on the provided context, including text and images.
2. Use Markdown: a numbered list for main items and bold for key terms.
3. For each item, state the **Key Term**, then add a '•' and a simple explanation.
4. If an image is provided, you MUST analyze it and refer to it in your explanation.
</rules>
<context>{context_str}</context>
<instructions>
Follow all rules to answer the user's question.
</instructions>
"""
    
    prompt_content = [{"type": "text", "text": prompt_text}]

    if b64_image:
        prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"})
    
    return prompt_content

# --- 3. GLOBAL AI SETUP ---
print("--- Initializing CogniVerse Backend ---")

print("Pre-loading local AI models into memory...")
final_rag_llm_local = OllamaLLM(model=FINAL_RESPONSE_MODEL_LOCAL, temperature=0.1)
condense_llm_local = OllamaLLM(model=QUESTION_CONDENSING_MODEL_LOCAL, temperature=0)
summary_llm_api = ChatGoogleGenerativeAI(model=SUMMARY_MODEL_API, google_api_key=GOOGLE_API_KEY, temperature=0)
print("✅ Local AI models loaded.")

print("Loading persistent stores...")
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = Chroma(
    collection_name="cogniverse-final-v9",
    embedding_function=embedding_function,
    persist_directory=str(VECTOR_STORE_PATH)
)
store = LocalFileStore(root_path=str(DOC_STORE_PATH))
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={'k': 10}
)

if not os.path.exists(DOC_STORE_PATH) or not os.listdir(DOC_STORE_PATH):
    # This block will run the first time to populate your databases
    print("⚠️ Persistent stores are empty. Populating now. This is a one-time process.")
    
    with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f: texts = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f: tables = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "image_paths.pkl", "rb") as f: image_paths = pickle.load(f)

    all_text_docs = [Document(page_content=t['text'], metadata={'source_page': t['source_page']}) for t in texts]
    all_text_docs.extend([Document(page_content=t['html'], metadata={'source_page': t['source_page']}) for t in tables])
    text_doc_ids = [str(uuid.uuid4()) for _ in all_text_docs]
    
    image_docs = [Document(page_content=p, metadata={'is_image': True}) for p in image_paths if os.path.exists(p)]
    image_ids = [str(uuid.uuid4()) for _ in image_docs]

    print("Generating text summaries via API...")
    text_summaries = []
    for i in tqdm(range(0, len(all_text_docs), 50), desc="Summarizing Text"):
        batch = [doc.page_content for doc in all_text_docs[i:i+50]]
        text_summaries.extend(generate_text_summaries_in_batch(summary_llm_api, batch))
        time.sleep(2)

    print("Generating image summaries via API...")
    image_summaries = []
    for i in tqdm(range(0, len(image_docs), 15), desc="Summarizing Images"):
        batch = [image_to_base64(doc.page_content) for doc in image_docs[i:i+15]]
        image_summaries.extend(generate_image_summaries_in_batch(summary_llm_api, batch))
        time.sleep(2)

    print("Populating persistent docstore...")
    encoded_text_docs = [(text_doc_ids[i], pickle.dumps(doc)) for i, doc in enumerate(all_text_docs)]
    encoded_image_docs = [(image_ids[i], pickle.dumps(doc)) for i, doc in enumerate(image_docs)]
    store.mset(encoded_text_docs)
    store.mset(encoded_image_docs)
    
    print("Populating persistent vectorstore with summaries...")
    text_summary_docs = [Document(page_content=s, metadata={id_key: text_doc_ids[i]}) for i, s in enumerate(text_summaries)]
    image_summary_docs = [Document(page_content=s, metadata={id_key: image_ids[i]}) for i, s in enumerate(image_summaries)]
    
    retriever.vectorstore.add_documents(documents=text_summary_docs + image_summary_docs)
    
    print("✅ All stores populated successfully.")
else:
    print("✅ Persistent stores already populated. Loading from disk.")

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
def deserialize_docs(docs: list) -> list[Document]:
    return [pickle.loads(doc) if isinstance(doc, bytes) else doc for doc in docs]

def condense_question(input: dict):
    if not input.get("chat_history"):
        return input["question"]
    condense_question_prompt = ChatPromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:""")
    return (condense_question_prompt | condense_llm_local | StrOutputParser()).invoke(input)


# In src/backend.py

def format_final_output(input_dict):
    # Get the text answer generated by the LLM
    final_answer = input_dict["text_answer"]
    
    # Find the first retrieved image document
    top_image_doc = next((doc for doc in input_dict["final_docs"] if doc.metadata.get('is_image')), None)

    # If an image was found, append its Markdown to the end of the answer
    if top_image_doc:
        b64_img = image_to_base64(top_image_doc.page_content)
        if b64_img:
            image_markdown = f"\n\n![Relevant Diagram](data:image/jpeg;base64,{b64_img})"
            final_answer += image_markdown

    return {
        "answer": final_answer,
        "image_paths": [doc.page_content for doc in input_dict["final_docs"] if doc.metadata.get('is_image', False)],
        "source_docs": [doc for doc in input_dict["final_docs"] if not doc.metadata.get('is_image', False)],
    }

# local llm -> failed to obey
# rag_chain_with_llm = (
#     RunnablePassthrough.assign(context=lambda x: format_for_final_prompt(x["final_docs"]))
#     .assign(final_prompt_content=lambda x: x["context"] + [{"type": "text", "text": f"\n\nQuestion: {x['standalone_question']}"}])
#     | ChatPromptTemplate.from_messages([("human", "{final_prompt_content}")])
#     | final_rag_llm_local
#     | StrOutputParser()
# )

# In src/backend.py, replace your entire rag_chain_with_llm definition

rag_chain_with_llm = (
    RunnablePassthrough.assign(
        # Step 1: Prepare the content (text and image parts)
        prompt_parts=lambda x: prepare_final_prompt_content(x["final_docs"])
    )
    | RunnablePassthrough.assign(
        # Step 2: Construct the final HumanMessage with the user's question at the end
        final_message=lambda x: HumanMessage(
            content=x["prompt_parts"] + [{"type": "text", "text": f"<question>\n{x['standalone_question']}\n</question>"}]
        )
    )
    # Step 3: Invoke the model with the single, correctly formatted message object
    | (lambda x: final_rag_llm_api.invoke([x["final_message"]]))
    | StrOutputParser()
)

gate = RunnableBranch(
    (lambda x: not x["final_docs"], lambda x: "Based on the provided textbook, I cannot answer this question."),
    rag_chain_with_llm,
)

chain = (
    RunnablePassthrough.assign(standalone_question=condense_question)
    | RunnablePassthrough.assign(hypothetical_document=hyde_chain)
    | RunnablePassthrough.assign(retrieved_docs_bytes=lambda x: retriever.invoke(x["hypothetical_document"]))
    | RunnablePassthrough.assign(retrieved_docs=lambda x: deserialize_docs(x["retrieved_docs_bytes"]))
    | RunnablePassthrough.assign(split_docs=lambda x: split_docs_by_type(x["retrieved_docs"]))
    | {
        "reranked_docs": (lambda x: re_ranker.compress_documents(query=x["standalone_question"], documents=x["split_docs"]["text_docs"]) if re_ranker else x["split_docs"]["text_docs"]),
        "image_docs": lambda x: x["split_docs"]["image_docs"],
        "question": itemgetter("question"),
        "standalone_question": itemgetter("standalone_question"),
    }
    | RunnablePassthrough.assign(final_docs=lambda x: x["reranked_docs"] + x["image_docs"])
    # NEW FINAL STEPS
    | RunnablePassthrough.assign(
        text_answer=gate, # 'gate' is your existing rag_chain_with_llm branch
    )
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
        answer = result.get('answer', "Sorry, I encountered an error.")
        sources = []
        for doc in result.get('source_docs', []):
            sources.append({
                "type": "text", 
                "content": doc.page_content, 
                "page": doc.metadata.get('source_page', 'N/A')
            })
        for path in result.get('image_paths', []):
            b64_img = image_to_base64(path)
            if b64_img:
                sources.append({
                    "type": "image", 
                    "content": f"data:image/jpeg;base64,{b64_img}", 
                    "filename": Path(path).name
                })
        
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
    
    
# # In src/backend.py (THE FINAL, WORKING, PERSISTENT VERSION)

# import os
# import uuid
# import base64
# import pickle
# from pathlib import Path
# from dotenv import load_dotenv
# import re
# from operator import itemgetter
# import time

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from tqdm import tqdm

# # --- Core LangChain Imports ---
# from langchain_core.documents import Document
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# # --- Specific LangChain Component Imports ---
# from langchain.storage import LocalFileStore
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import OllamaLLM
# from langchain_cohere import CohereRerank
# from langchain_google_genai import ChatGoogleGenerativeAI

# # --- 1. CONFIGURATION ---
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
# VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"
# DOC_STORE_PATH = PROJECT_ROOT / "doc_store" 

# FINAL_RESPONSE_MODEL_LOCAL = "llava"
# QUESTION_CONDENSING_MODEL_LOCAL = "phi3:mini"
# SUMMARY_MODEL_API = "gemini-1.5-flash"

# # --- 2. HELPER FUNCTIONS ---
# def parse_llm_summaries(llm_output: str, expected_count: int) -> list[str]:
#     lines = llm_output.strip().split('\n')
#     summary_map = {}
#     for line in lines:
#         match = re.match(r'^(\d+)[.:\s]\s*(.*)', line.strip())
#         if match:
#             num, summary_text = int(match.group(1)), match.group(2).strip()
#             if 1 <= num <= expected_count:
#                 summary_map[num] = summary_text
#     return [summary_map.get(i, "[SUMMARY_GENERATION_FAILED]") for i in range(1, expected_count + 1)]

# def generate_text_summaries_in_batch(llm, text_batch: list[str]) -> list[str]:
#     if not text_batch: return []
#     prompt_text = f"""<instructions>
# You are an expert at summarizing academic text for a search index. You will be given a batch of {len(text_batch)} text chunks from a textbook.
# Your task is to provide a concise, one-sentence summary for EACH text chunk.
# - You MUST number your summaries starting from 1.
# - The summary on line N MUST correspond to CHUNK N.
# - The summary MUST capture the main keywords, definitions, and concepts in the chunk.
# - Do NOT add any extra text, conversation, or apologies.
# - If a chunk is nonsensical or too short to summarize, output the number and the exact text: [SUMMARY_GENERATION_FAILED]
# </instructions>
# <example>
# --- CHUNK 1 ---
# 4.1.1 Public, Private, and Hybrid Clouds. The concept of cloud computing has evolved from cluster, grid, and utility computing.
# --- CHUNK 2 ---
# A public cloud is owned by the public and accessible by any user who has paid for the service.
# Your output MUST be in this exact format:
# 1. An introduction to cloud computing, its evolution from cluster and grid computing, and the three main types: public, private, and hybrid.
# 2. A definition of a public cloud as a service owned by providers and accessible on a pay-per-use basis.
# </example>
# Here are the {len(text_batch)} text chunks:
# """
#     for i, text in enumerate(text_batch):
#         prompt_text += f"--- CHUNK {i+1} ---\n{text}\n\n"
#     try:
#         msg = llm.invoke(prompt_text)
#         return parse_llm_summaries(msg.content, len(text_batch))
#     except Exception as e:
#         print(f"Error in text summarization: {e}")
#         return ["[SUMMARY_API_ERROR]"] * len(text_batch)

# def generate_image_summaries_in_batch(llm, image_batch_b64: list) -> list[str]:
#     if not image_batch_b64: return []
#     prompt_text = f"""<instructions>
# You are an expert at analyzing textbook figures. You will be given {len(image_batch_b64)} images.
# Your task is to provide a concise, one-sentence summary for EACH image.
# - You MUST number your summaries starting from 1.
# - The summary on line N MUST correspond to IMAGE N.
# - The summary MUST capture the main keywords and concepts shown in the image.
# - Do NOT add any extra text, conversation, or apologies.
# - If you cannot analyze an image, output the number and the exact text: [SUMMARY_GENERATION_FAILED]
# </instructions>
# <example>
# IMAGE 1 is a diagram.
# IMAGE 2 is a graph.
# Your output MUST be in this exact format given below, and NOT A WORD MORE or LESS:
# 1. A diagram illustrating the client-server architecture in cloud computing.
# 2. A graph showing the performance scaling of parallel processing tasks.
# </example>
# Here are the {len(image_batch_b64)} images:
# """
#     prompt_content = [{"type": "text", "text": prompt_text}]
#     for b64 in image_batch_b64:
#         prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"})
#     try:
#         msg = llm.invoke([HumanMessage(content=prompt_content)])
#         return parse_llm_summaries(msg.content, len(image_batch_b64))
#     except Exception as e:
#         print(f"Error in image summarization: {e}")
#         return ["[SUMMARY_API_ERROR]"] * len(image_batch_b64)

# def split_docs_by_type(docs):
#     text_docs, image_docs = [], []
#     for doc in docs:
#         if doc.metadata.get('is_image', False): image_docs.append(doc)
#         else: text_docs.append(doc)
#     return {"text_docs": text_docs, "image_docs": image_docs}

# def image_to_base64(image_path):
#     try:
#         with open(image_path, "rb") as img_file: return base64.b64encode(img_file.read()).decode('utf-8')
#     except Exception: return None

# def format_for_final_prompt(docs):
#     prompt_content = [{"type": "text", "text": "You are an expert university study buddy. Your primary directive is to act as a tutor and give a justifiable, well-written, and easy-to-read answer based STRICTLY AND ONLY on the provided context, which may include text, tables, and images.\n\n**Instructions:**\n1.  **Synthesize, Do Not Just List:** Read all the provided context documents. Weave the information into a **single, cohesive, flowing answer.**\n2.  **Analyze and Explain Images/Tables:** If images or tables are present, **analyze their content** and explain what they illustrate in relation to the user's question.\n3.  **Format for Readability:** Use Markdown for formatting (bold, lists, etc.).\n4.  **Strictly Adhere to Context:** If the context does not contain enough information, you MUST respond with exactly this phrase: \"Based on the provided textbook, I cannot answer this question.\" and not a word more.\n\n--- CONTEXT START ---"}]
#     for doc in docs:
#         if doc.metadata.get('is_image', False):
#             b64_image = image_to_base64(doc.page_content)
#             if b64_image: prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"})
#         else:
#             source_page = doc.metadata.get('source_page', 'N/A')
#             prompt_content.append({"type": "text", "text": f"\n[Text/Table from Page {source_page}]:\n{doc.page_content}"})
#     prompt_content.append({"type": "text", "text": "\n--- CONTEXT END ---\n"})
#     return prompt_content

# # --- 3. GLOBAL AI SETUP ---
# print("--- Initializing CogniVerse Backend ---")

# print("Pre-loading local AI models into memory...")
# final_rag_llm_local = OllamaLLM(model=FINAL_RESPONSE_MODEL_LOCAL, temperature=0.1)
# condense_llm_local = OllamaLLM(model=QUESTION_CONDENSING_MODEL_LOCAL, temperature=0)
# summary_llm_api = ChatGoogleGenerativeAI(model=SUMMARY_MODEL_API, google_api_key=GOOGLE_API_KEY, temperature=0)
# print("✅ Local AI models loaded.")

# print("Loading persistent stores...")
# embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
# vectorstore = Chroma(
#     collection_name="cogniverse-final-v9",
#     embedding_function=embedding_function,
#     persist_directory=str(VECTOR_STORE_PATH)
# )
# store = LocalFileStore(root_path=str(DOC_STORE_PATH))
# id_key = "doc_id"

# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={'k': 10}
# )

# if not os.path.exists(DOC_STORE_PATH) or not os.listdir(DOC_STORE_PATH):
#     print("⚠️ Persistent stores are empty. Populating now. This is a one-time process.")
    
#     with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f: texts = pickle.load(f)
#     with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f: tables = pickle.load(f)
#     with open(PROCESSED_DATA_DIR / "image_paths.pkl", "rb") as f: image_paths = pickle.load(f)

#     all_text_docs = [Document(page_content=t['text'], metadata={'source_page': t['source_page']}) for t in texts]
#     all_text_docs.extend([Document(page_content=t['html'], metadata={'source_page': t['source_page']}) for t in tables])
#     text_doc_ids = [str(uuid.uuid4()) for _ in all_text_docs]
    
#     image_docs = [Document(page_content=p, metadata={'is_image': True}) for p in image_paths if os.path.exists(p)]
#     image_ids = [str(uuid.uuid4()) for _ in image_docs]

#     print("Generating text summaries via API...")
#     TEXT_BATCH_SIZE = 50
#     text_contents = [doc.page_content for doc in all_text_docs]
#     text_summaries = []
#     for i in tqdm(range(0, len(text_contents), TEXT_BATCH_SIZE), desc="Summarizing Text Chunks"):
#         batch_text = text_contents[i:i+TEXT_BATCH_SIZE]
#         text_summaries.extend(generate_text_summaries_in_batch(summary_llm_api, batch_text))
#         time.sleep(2)

#     print("Generating image summaries via API...")
#     IMAGE_BATCH_SIZE = 15
#     image_base64s = [image_to_base64(doc.page_content) for doc in image_docs]
#     image_summaries = []
#     for i in tqdm(range(0, len(image_base64s), IMAGE_BATCH_SIZE), desc="Summarizing Images"):
#         batch_b64 = image_base64s[i:i+IMAGE_BATCH_SIZE]
#         image_summaries.extend(generate_image_summaries_in_batch(summary_llm_api, batch_b64))
#         time.sleep(2)

#     print("Populating persistent docstore...")
#     encoded_text_docs = [(text_doc_ids[i], pickle.dumps(doc)) for i, doc in enumerate(all_text_docs)]
#     encoded_image_docs = [(image_ids[i], pickle.dumps(doc)) for i, doc in enumerate(image_docs)]
#     store.mset(encoded_text_docs)
#     store.mset(encoded_image_docs)
    
#     print("Populating persistent vectorstore with summaries...")
#     text_summary_docs = [Document(page_content=s, metadata={id_key: text_doc_ids[i]}) for i, s in enumerate(text_summaries)]
#     image_summary_docs = [Document(page_content=s, metadata={id_key: image_ids[i]}) for i, s in enumerate(image_summaries)]
    
#     retriever.vectorstore.add_documents(documents=text_summary_docs)
#     retriever.vectorstore.add_documents(documents=image_summary_docs)
    
#     print("✅ All stores populated successfully.")
# else:
#     print("✅ Persistent stores already populated. Loading from disk.")

# re_ranker = None
# if COHERE_API_KEY and "YOUR_TRIAL_API_KEY" not in COHERE_API_KEY:
#     print("✅ Cohere re-ranker is enabled.")
#     re_ranker = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=3, model="rerank-english-v3.0")
# else:
#     print("⚠️ Cohere re-ranker is disabled.")

# hyde_prompt_template = "Please generate a concise, one-paragraph, textbook-style answer for the following question to help find relevant documents.\nQuestion: {question}\nHypothetical Answer:"
# hyde_prompt = PromptTemplate(input_variables=["question"], template=hyde_prompt_template)
# hyde_chain = hyde_prompt | condense_llm_local | StrOutputParser()

# # --- 4. THE CONVERSATIONAL RAG CHAIN ---
# def deserialize_docs(docs: list) -> list[Document]:
#     # THIS IS THE "UNPACKING" STATION
#     return [pickle.loads(doc) if isinstance(doc, bytes) else doc for doc in docs]

# def condense_question(input: dict):
#     if not input.get("chat_history"):
#         return input["question"]
#     condense_question_prompt = ChatPromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
# Chat History: {chat_history}
# Follow Up Input: {question}
# Standalone question:""")
#     return (condense_question_prompt | condense_llm_local | StrOutputParser()).invoke(input)

# def format_final_output(input_dict):
#     return {
#         "answer": input_dict["answer"],
#         "image_paths": [doc.page_content for doc in input_dict["final_docs"] if doc.metadata.get('is_image', False)],
#         "source_docs": [doc for doc in input_dict["final_docs"] if not doc.metadata.get('is_image', False)],
#     }

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
#     | RunnablePassthrough.assign(hypothetical_document=hyde_chain)
#     | RunnablePassthrough.assign(retrieved_docs_bytes=lambda x: retriever.invoke(x["hypothetical_document"]))
#     # BUG FIX: Add the "unpacking" station right after retrieval
#     | RunnablePassthrough.assign(retrieved_docs=lambda x: deserialize_docs(x["retrieved_docs_bytes"]))
#     | RunnablePassthrough.assign(split_docs=lambda x: split_docs_by_type(x["retrieved_docs"]))
#     | {
#         "reranked_docs": (lambda x: re_ranker.compress_documents(query=x["standalone_question"], documents=x["split_docs"]["text_docs"]) if re_ranker else x["split_docs"]["text_docs"]),
#         "image_docs": lambda x: x["split_docs"]["image_docs"],
#         "question": itemgetter("question"),
#         "standalone_question": itemgetter("standalone_question"),
#     }
#     | RunnablePassthrough.assign(final_docs=lambda x: x["reranked_docs"] + x["image_docs"])
#     | RunnablePassthrough.assign(answer=gate)
#     | format_final_output
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
#     if not user_query:
#         return jsonify({"error": "No question provided"}), 400
#     try:
#         result = chain.invoke({"question": user_query, "chat_history": chat_history})
#         answer = result.get('answer', "Sorry, I encountered an error.")
#         sources = []
#         for doc in result.get('source_docs', []):
#             sources.append({
#                 "type": "text", 
#                 "content": doc.page_content, 
#                 "page": doc.metadata.get('source_page', 'N/A')
#             })
#         for path in result.get('image_paths', []):
#             b64_img = image_to_base64(path)
#             if b64_img:
#                 sources.append({
#                     "type": "image", 
#                     "content": f"data:image/jpeg;base64,{b64_img}", 
#                     "filename": Path(path).name
#                 })
        
#         chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
#         if len(chat_history) > 10:
#             chat_history = chat_history[-10:]
        
#         return jsonify({"answer": answer, "sources": sources})
#     except Exception as e:
#         print(f"ERROR in /ask endpoint: {e}")
#         return jsonify({"error": "An internal error occurred."}), 500

# @app.route('/reset', methods=['POST'])
# def reset():
#     global chat_history
#     chat_history = []
#     return jsonify({"message": "Chat history reset successfully"})

# if __name__ == '__main__':
#     print("🚀 CogniVerse backend is running at http://127.0.0.1:5000")
#     app.run(host='0.0.0.0', port=5000)
