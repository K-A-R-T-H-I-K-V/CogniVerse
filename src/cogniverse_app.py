import os
import uuid
import base64
import pickle
from pathlib import Path
from dotenv import load_dotenv

# --- Core LangChain Imports ---
# These are the fundamental building blocks of our application.
from langchain_core.documents import Document # The standard format for a piece of text with metadata.
from langchain_core.messages import HumanMessage, AIMessage # Standard message types for chat models.
from langchain_core.prompts import ChatPromptTemplate # For creating robust prompts for chat models.
from langchain_core.output_parsers import StrOutputParser # A simple parser to get the string content from an LLM's response.
from langchain_core.runnables import RunnablePassthrough # A utility to pass inputs through a chain unchanged.

# --- Specific LangChain Component Imports ---
# These are the specific tools we've chosen for each step of our advanced pipeline.
from langchain.storage import InMemoryStore # A simple, dictionary-like key-value store that lives in RAM.
from langchain.retrievers.multi_vector import MultiVectorRetriever # The state-of-the-art retriever for multimodal RAG.
from langchain_chroma import Chroma # Our chosen persistent vector store.
from langchain_huggingface import HuggingFaceEmbeddings # The modern way to use Hugging Face embedding models.
from langchain_ollama import OllamaLLM # The modern way to interface with local Ollama LLMs.
from langchain_cohere import CohereRerank # The powerful re-ranking model.

# For showing progress bars during the initial setup.
from tqdm import tqdm

# --- 1. CONFIGURATION and LOADING ---
# This block is identical to the data_processor script, ensuring consistency.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Paths for our pre-processed data and where we'll save/load the persistent ChromaDB vector store.
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store_chroma"
IMAGE_DIR = PROCESSED_DATA_DIR / "images"

# We define which local Ollama LLMs to use for different tasks. This allows flexibility.
# For simple summarization, a small, fast model is fine. For the final answer, we need the powerful multimodal model.
TEXT_SUMMARY_MODEL = "phi3:mini"
IMAGE_SUMMARY_MODEL = "llava"
FINAL_RESPONSE_MODEL = "llava"
QUESTION_CONDENSING_MODEL = "phi3:mini"

# --- 2. HELPER FUNCTIONS ---

def image_to_base64(image_path):
    """Converts an image file to a base64 string. This format is required to embed images directly into a prompt for multimodal LLMs."""
    try:
        with open(image_path, "rb") as img_file:
            # .read() gets the binary content of the image.
            # base64.b64encode() converts this binary to base64 bytes.
            # .decode('utf-8') converts the base64 bytes into a text string that can be sent in an API call.
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def generate_summary(llm, content, prompt_template):
    """Generates a text summary for a given piece of text or table content."""
    # This function builds a simple LangChain Expression Language (LCEL) chain: Prompt -> LLM -> String Output.
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    # .invoke() executes the chain.
    return chain.invoke({"element": content})

def generate_image_summary(llm, image_base64):
    """Generates a text summary for a base64-encoded image using a multimodal LLM like LLaVA."""
    # To talk to a multimodal model, we send a list of content parts.
    # One part is the text instruction, and the other part is the image data.
    msg = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Provide a concise, one-sentence summary of this image's content. This summary will be used as a searchable index."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                ]
            )
        ]
    )
    return msg.content

def format_docs_for_display(docs, image_paths):
    """A utility function to pretty-print the retrieved sources for the user in the terminal."""
    # This is purely for user experience in our command-line app.
    formatted_string = ""
    # We first handle all the non-image documents.
    text_docs = [doc for doc in docs if not doc.metadata.get('is_image', False)]
    for i, doc in enumerate(text_docs):
        source_page = doc.metadata.get('source_page', 'N/A')
        formatted_string += f"\n--- Retrieved Text/Table {i+1} (Source: Page {source_page}) ---\n"
        formatted_string += doc.page_content + "\n"
    
    # Then we list the image files that were retrieved.
    if image_paths:
        formatted_string += "\n--- Retrieved Images ---"
        for path in image_paths:
            # We use Path(path).name to show just the filename (e.g., 'page_10_img_1.jpg').
            formatted_string += f"\n- {Path(path).name} (Path: {path})"
    return formatted_string

# --- 3. MAIN APPLICATION LOGIC ---

def main():
    print("--- Initializing CogniVerse Multimodal RAG Application ---")

    # --- Step A: Load Pre-processed Data from Disk ---
    try:
        # We use pickle.load() to instantly load our lists of text and table dictionaries from the files created by data_processor.py.
        # "rb" means "read in binary mode."
        with open(PROCESSED_DATA_DIR / "texts.pkl", "rb") as f:
            texts = pickle.load(f)
        with open(PROCESSED_DATA_DIR / "tables.pkl", "rb") as f:
            tables = pickle.load(f)
        # We get a list of all image file paths and sort them to ensure a consistent processing order.
        image_paths = sorted([str(p) for p in IMAGE_DIR.glob("*.jpg")])
    except FileNotFoundError:
        print("\n❌ Pre-processed data not found. Please run `python src/data_processor.py` first.")
        return

    # --- Step B: Initialize LLMs ---
    # We create instances of our Ollama LLMs. Setting temperature=0 makes the summary generation
    # more deterministic and factual. A slightly higher temperature for the final answer allows for more natural language.
    text_llm = OllamaLLM(model=TEXT_SUMMARY_MODEL, temperature=0)
    image_llm = OllamaLLM(model=IMAGE_SUMMARY_MODEL, temperature=0)
    final_rag_llm = OllamaLLM(model=FINAL_RESPONSE_MODEL, temperature=0.1)
    condense_llm = OllamaLLM(model=QUESTION_CONDENSING_MODEL, temperature=0)

    # --- Step C: Setup the Multi-Vector Retriever ---
    print("Setting up the Multi-Vector Retriever...")
    
    # We initialize ChromaDB. It will automatically create the database files in the `vector_store_chroma`
    # directory if they don't exist, or load them if they do. This makes our data persistent.
    vectorstore = Chroma(
        collection_name="cogniverse_summaries",
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=str(VECTOR_STORE_PATH)
    )

    # This is a simple in-memory Python dictionary that will hold our original, full-sized documents.
    store = InMemoryStore()
    id_key = "doc_id" # This is the metadata key we'll use to link summaries in the vector store to original docs in the docstore.
    
    # This is the core of the architecture. It's initialized with our two storage components.
    # It knows that when it finds a summary vector in the `vectorstore`, it should use the `doc_id`
    # from that summary's metadata to fetch the corresponding full document from the `docstore`.
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # --- Step D: Populate the Retriever (The Slow, One-Time Process) ---
    # We check if the persistent ChromaDB vector store already has data by checking for any IDs.
    # If it does, we can skip this entire slow summarization and embedding process.
    if not vectorstore.get()['ids']:
        print("Vector store is empty. Populating with summaries and linking to original data...")
        
        # Define a cache file to save our progress
        summary_cache_file = PROCESSED_DATA_DIR / "summaries.pkl"
        
        # Check if we have already made some progress
        if summary_cache_file.exists():
            with open(summary_cache_file, "rb") as f:
                cached_summaries = pickle.load(f)
            print(f"Loaded {len(cached_summaries['texts'])} cached text summaries and {len(cached_summaries['tables'])} table summaries.")
        else:
            cached_summaries = {"texts": [], "tables": [], "images": []}

        # Define batch size
        BATCH_SIZE = 10
        
        # --- Process Texts in Batches ---
        if len(cached_summaries["texts"]) < len(texts):
            text_prompt = "Provide a concise, one-sentence summary of the following text that captures its main keywords and concepts. This summary will be used for a search index. Element: {element}"
            for i in tqdm(range(len(cached_summaries["texts"]), len(texts), BATCH_SIZE), desc="Summarizing Texts"):
                batch = texts[i:i+BATCH_SIZE]
                summaries = [generate_summary(text_llm, t['text'], text_prompt) for t in batch]
                cached_summaries["texts"].extend(summaries)
                with open(summary_cache_file, "wb") as f:
                    pickle.dump(cached_summaries, f)
        
        # --- Process Tables in Batches ---
        if len(cached_summaries["tables"]) < len(tables):
            table_prompt = "Provide a concise, one-sentence summary of the following table content that captures its main keywords and concepts. This summary will be used for a search index. Element: {element}"
            for i in tqdm(range(len(cached_summaries["tables"]), len(tables), BATCH_SIZE), desc="Summarizing Tables"):
                batch = tables[i:i+BATCH_SIZE]
                summaries = [generate_summary(text_llm, t['html'], table_prompt) for t in batch]
                cached_summaries["tables"].extend(summaries)
                with open(summary_cache_file, "wb") as f:
                    pickle.dump(cached_summaries, f)
        
        # --- Process Images in Batches ---
        valid_image_paths = [p for p in image_paths if image_to_base64(p) is not None]
        if len(cached_summaries["images"]) < len(valid_image_paths):
            for i in tqdm(range(len(cached_summaries["images"]), len(valid_image_paths), BATCH_SIZE), desc="Summarizing Images"):
                batch_paths = valid_image_paths[i:i+BATCH_SIZE]
                batch_b64 = [image_to_base64(p) for p in batch_paths]
                summaries = [generate_image_summary(image_llm, b64) for b64 in batch_b64]
                cached_summaries["images"].extend(summaries)
                with open(summary_cache_file, "wb") as f:
                    pickle.dump(cached_summaries, f)

        text_summaries = cached_summaries["texts"]
        table_summaries = cached_summaries["tables"]
        image_summaries = cached_summaries["images"]
        
        # The rest of the population logic is the same...
        text_ids = [str(uuid.uuid4()) for _ in texts]
        table_ids = [str(uuid.uuid4()) for _ in tables]
        image_ids = [str(uuid.uuid4()) for _ in valid_image_paths]
        
        original_texts = [Document(page_content=t['text'], metadata={'source_page': t['source_page']}) for t in texts]
        retriever.docstore.mset(list(zip(text_ids, original_texts)))
        
        original_tables = [Document(page_content=t['html'], metadata={'source_page': t['source_page']}) for t in tables]
        retriever.docstore.mset(list(zip(table_ids, original_tables)))
        
        original_images = [Document(page_content=p, metadata={'is_image': True}) for p in valid_image_paths]
        retriever.docstore.mset(list(zip(image_ids, original_images)))

        summary_docs = []
        for i, summary in enumerate(text_summaries):
            summary_docs.append(Document(page_content=summary, metadata={id_key: text_ids[i]}))
        for i, summary in enumerate(table_summaries):
            summary_docs.append(Document(page_content=summary, metadata={id_key: table_ids[i]}))
        for i, summary in enumerate(image_summaries):
            summary_docs.append(Document(page_content=summary, metadata={id_key: image_ids[i]}))
        
        retriever.vectorstore.add_documents(summary_docs)
        vectorstore.persist()
        print("✅ Retriever populated and vector store persisted.")
    else:
        print("✅ Vector store already populated. Loading from disk.")

    # --- Step E: Setup the Re-ranker ---
    re_ranker = None
    if COHERE_API_KEY and "YOUR_TRIAL_API_KEY" not in COHERE_API_KEY:
        print("✅ Cohere re-ranker is enabled.")
        # top_n=4 means it will return the best 4 documents from the initial search.
        re_ranker = CohereRerank(
            cohere_api_key=COHERE_API_KEY, top_n=4, model="rerank-english-v3.0"
        )
    else:
        print("⚠️ Cohere re-ranker is disabled (API key not provided).")

    # --- Step F: Define the Final Conversational RAG Chain ---
    # This is where we wire everything together using LangChain Expression Language (LCEL).
    
    # This sub-chain handles chat history. It takes the history and a new question,
    # and uses a small LLM to rephrase it into a standalone question.
    condense_question_prompt = ChatPromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")
    
    condense_question_chain = condense_question_prompt | condense_llm | StrOutputParser()

    # A simple router function. If chat history exists, it runs the condensing chain.
    # If not, it just passes the original question through.
    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_question_chain
        else:
            return input["question"]

    # This function takes the final, re-ranked documents and prepares them for the
    # multimodal LLM prompt. It converts image paths to base64 ONLY at this last moment.
    def format_for_final_prompt(docs):
        prompt_content = []
        # This is our new, fortified prompt, directly inserted here.
        prompt_content.append({"type": "text", "text": """You are an expert university study buddy. Your primary directive is to act as a tutor and give a justifiable, well-written, and easy-to-read answer based STRICTLY AND ONLY on the provided context, which may include text, tables, and images.

**Instructions:**
1.  **Synthesize, Do Not Just List:** Read all the provided context documents. Weave the information into a **single, cohesive, flowing answer.** Do not list information from different documents separately. Your answer should read like a single, well-written explanation from an expert tutor.
2.  **Analyze and Explain Images/Tables:** If images or tables are present in the context, do not just mention them. **Analyze their content** and explain what they illustrate in relation to the user's question.
3.  **Format for Readability:** Use Markdown for formatting. Use headings and subheadings. **Bold** key terms and definitions. Use bullet points for lists.
4.  **Strictly Adhere to Context:** If the context does not contain enough information to answer the question, you MUST respond with exactly this phrase: "Based on the provided textbook, I cannot answer this question." and not a word more. Do not use any outside knowledge.

--- CONTEXT START ---"""})

        for doc in docs:
            # We check the metadata to see if this document is an image.
            if doc.metadata.get('is_image', False):
                # If it's an image document, its `page_content` is the file path.
                # We call our helper to convert it to base64 for the LLM.
                image_base64 = image_to_base64(doc.page_content)
                if image_base64:
                    prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"})
            else:
                # If it's not an image, it's text or a table.
                source_page = doc.metadata.get('source_page', 'N/A')
                prompt_content.append({"type": "text", "text": f"\n[Text/Table from Page {source_page}]:\n{doc.page_content}"})

        prompt_content.append({"type": "text", "text": "\n--- CONTEXT END ---\n"})
        return prompt_content
    
    # This is the main chain. The `|` symbol pipes the output of one component to the input of the next.
    # `RunnablePassthrough.assign` is a powerful LCEL tool to add new keys to the dictionary as it flows through the chain.
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
                | final_rag_llm
                | StrOutputParser()
            ),
            # *** THIS IS HOW WE OUTPUT THE IMAGE PATHS ***
            # We add a new key to our final output dictionary.
            # This lambda function filters the re-ranked documents and returns only the file paths of the images.
            "image_paths": lambda x: [doc.page_content for doc in x["reranked_docs"] if doc.metadata.get('is_image', False)],
            # We do the same for text sources for clean citation.
            "source_docs": lambda x: [doc for doc in x["reranked_docs"] if not doc.metadata.get('is_image', False)],
        }
    )

    print("\n🚀 CogniVerse is ready! Ask your multimodal questions.")
    print("-" * 50)
    
    # --- Step G: Interactive Application Loop (Updated) ---
    chat_history = []
    while True:
        try:
            user_query = input("\nAsk a question: ")
            if user_query.lower() == 'exit':
                print("Goodbye! Happy studying.")
                break
            
            # Invoking the chain gives us a dictionary with 'answer', 'image_paths', and 'source_docs'.
            result = chain.invoke({"question": user_query, "chat_history": chat_history})
            
            # We append the user's query and the AI's TEXT answer to the history.
            chat_history.extend([
                HumanMessage(content=user_query),
                AIMessage(content=result["answer"]),
            ])

            print("\n--- Answer ---")
            print(result["answer"])
            
            # *** NEW OUTPUT SECTION TO DISPLAY IMAGE PATHS ***
            if result["image_paths"]:
                print("\n--- Relevant Images Found ---")
                for path in result["image_paths"]:
                    print(f"- {Path(path).name} (Path: {path})")
                print("-----------------------------")

            print("\n--- Retrieved Text Sources ---")
            # We now use our display helper to show the text/table sources.
            print(format_docs_for_display(result["source_docs"], []))
            print("----------------------------")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()