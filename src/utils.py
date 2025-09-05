from typing import List, Dict
import ollama
import re

# --- Core LangChain components ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

# --- For Embeddings and the LLM ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# def load_and_chunk_document(pdf_path, chunk_size=1000, chunk_overlap=400):
#     """
#     Loads a PDF document and splits it into smaller, manageable chunks.
#     """
#     print(f"Loading document from: {pdf_path}")
#     loader = PyMuPDFLoader(pdf_path) # It creates an instance of PyMuPDFLoader, a specific Document Loader from LangChain.
#     documents = loader.load() # The .load() method iterates through the PDF, page by page. For each page, it extracts the text and creates a LangChain Document object. Each Document contains two main parts: page_content (the actual text from the page) and metadata (a dictionary containing information like the source file path and page number)
    
#     print(f"Splitting {len(documents)} pages into chunks...")
#     # It tries to split on logical separators first (like paragraph breaks) before resorting to just cutting words
#     text_splitter = RecursiveCharacterTextSplitter( # By default, it tries to split by ["\n\n", "\n", " ", ""]. This means it first looks for double newlines (paragraph breaks), then single newlines, then spaces.
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap, # the overlap ensures the full sentence is present in at least one of them, preventing the model from getting a fragmented view
#         length_function=len,
#     )
#     chunks = text_splitter.split_documents(documents) # Importantly, it preserves the metadata, so each new chunk still knows which page and source file it came from. This is essential for providing citations later.
#     print(f"Successfully created {len(chunks)} text chunks.")
#     return chunks


# Step 1: Document Layout Analysis (DLA)
# Step 2: Element Classification
# Step 3: From Elements to Structure-Aware Chunks

# This is a helper function whose sole purpose is to convert the structured output from the unstructured library into a clean, single Markdown document. Markdown is a simple text format that uses characters like # for headings and * for list items, which is perfect for representing a document's structure.
def elements_to_markdown(documents):
    """
    Converts a list of LangChain Document objects (from UnstructuredLoader)
    to a single Markdown string, respecting the element types in the metadata.
    """
    markdown_string = ""
    for doc in documents:
        # Unstructured detects the element type (e.g., Title, NarrativeText, ListItem). Loops through each 'Element' that 'unstructured' extracted.
        # This line checks the type of the current element. 'unstructured' classifies
        # text blocks into types like 'Title', 'ListItem', 'NarrativeText', etc.
        category = doc.metadata.get('category', '')
        if category == 'Title':
            # Convert titles to Markdown headers. You can adjust the number of '#'
            # based on element metadata if available (e.g., header level)
            # If the element is a title, we format it as a top-level Markdown header
            # by adding a '#' in front of it, followed by two newlines for spacing.
            markdown_string += f"# {doc.page_content}\n\n"
        elif category == 'ListItem':
            # This uses a regular expression to remove any pre-existing bullet points
            # ('-', '*', '•') that 'unstructured' might have included, ensuring clean formatting.
            clean_text = re.sub(r'^(?:\s*[-*•]\s*)+', '', doc.page_content)
            markdown_string += f"* {clean_text}\n"
        else:
            # For narrative text and other elements, just add them as paragraphs
            markdown_string += f"{doc.page_content}\n\n"
    return markdown_string

def load_and_chunk_document(pdf_path):
    """
    Loads a PDF using unstructured, converts it to Markdown, and then splits it
    based on the headers. This creates semantically coherent chunks.
    """
    print(f"Loading document with Unstructured: {pdf_path}")
    # The "elements" mode is key to getting the structured output
    # Creates an instance of the UnstructuredPDFLoader.
    # mode="elements": This tells the loader to not return a single block of text,
    # but a list of classified structural elements (Title, ListItem, etc.).
    # strategy="hi_res": This uses a high-resolution model that is better at
    # analyzing document layouts, especially for multi-column PDFs
    loader = UnstructuredPDFLoader(pdf_path, mode="elements", strategy="hi_res")
    documents = loader.load() # # Executes the loading process, returning the list of 'Element' objects.

    print("Converting extracted elements to a single Markdown document...")
    # Calls our helper function to convert the list of elements into one cohesive
    # Markdown-formatted string.
    markdown_text = elements_to_markdown(documents)
    
    # This defines which Markdown headers the splitter should use as separation points.
    # We're telling it that any line starting with '#' is a "Header 1" and should
    # mark the beginning of a new chunk.
    headers_to_split_on = [
        ("#", "Header 1"),
    ]

    print("Splitting the Markdown document based on headers...")
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # This is where the splitting happens. It takes the Markdown text and creates
    # chunks, where each chunk contains a header and all the text that follows it,
    # up until the next header.
    chunks = markdown_splitter.split_text(markdown_text)

    # We should also add the source page metadata back to each chunk if possible
    # For now, we'll proceed with the text content which is the most important part.
    print(f"Successfully created {len(chunks)} structure-aware chunks.")
    return chunks
    
def create_vector_store(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Converts text chunks into numerical vectors (embeddings) and stores them
    in a FAISS vector store for fast searching.
    """
    print("Creating text embeddings and building the vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name) # This model is a variant of BERT, fine-tuned specifically to produce high-quality sentence embeddings. When you pass text to it, the model performs a forward pass, converting the text into a dense vector (a 384-dimensional vector in this case) that numerically represents its semantic meaning
    
    vector_store = FAISS.from_documents(chunks, embeddings) # For each chunk's page_content, it calls the embeddings model to get its vector representation. It builds a data structure, or index, that allows for incredibly fast Approximate Nearest Neighbor (ANN) searches
    print("Vector store created successfully.")
    return vector_store

def create_conversational_rag_chain(vector_store, llm_model_name="phi3:mini", cohere_api_key=None):
    """
    Creates the full Conversational RAG chain with memory and re-ranking.
    """
    print("Setting up the Conversational RAG pipeline...")
    # 1. Define the LLM
    llm = OllamaLLM(model=llm_model_name) # It creates a client to communicate with your local LLM.
    
    # 2. Define the Retriever
    base_retriever = vector_store.as_retriever(search_kwargs={'k': 10}) # 1. Retrieve more documents.  This turns our vector store into a "retriever." Its only job is to listen for a question, convert that question into a vector, and quickly find the k=10 most similar text chunks from the vector store.
    # Its core job is to accept a string query, use the embedding model to turn that query into a vector, and then use the vector store's similarity search function (in this case, FAISS's ANN search) to find the top k most similar documents

    if cohere_api_key:
        print("✅ Cohere re-ranker is enabled.")
        reranker = CohereRerank(
            cohere_api_key=cohere_api_key, top_n=3, model="rerank-english-v3.0"
        )
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )
    else:
        print("⚠️ Cohere re-ranker is disabled (API key not provided).")
        final_retriever = base_retriever

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer'
    )
    
    # 3. Define the Prompt Template-
    prompt_template = """You are an expert university study buddy. Your goal is to give a justifiable, well-written, and easy-to-read answer based ONLY on the provided textbook context.

**Instructions:**
1.  **Synthesize, Do Not Just List:** Read all the provided context. Weave the information into a single, cohesive answer. Do not list information from different documents separately. Your answer should flow like a single, well-written explanation.
2.  **Use Textbook Structure:** If the context contains headings or subheadings, use them to structure your answer.
3.  **Format for Readability:** Use Markdown for formatting. **Bold** key terms and definitions. Use bullet points for lists.
4.  **Strictly Adhere to Context:** If the context does not contain the answer, you MUST respond with exactly this phrase: "Based on the provided textbook, I cannot answer this question." and not a word more. Do not use any outside knowledge.

**Context from Textbook:**
{context}

**Question:**
{question}

**Answer:**
"""
    
    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # # 4. Create the RAG Chain
    # # | symbol "pipes" the output of one step into the input of the next
    # rag_chain = (
    #     {"context": final_retriever, "question": RunnablePassthrough()} #{"context": retriever, "question": RunnablePassthrough()}: When you ask a question, this step simultaneously (a) sends the question to the retriever to fetch context and (b) passes the original question through untouched. {'context': [doc1, doc2, ...], 'question': 'your question'}.
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # --- Create the ConversationalRetrievalChain ---
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=final_retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer',
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    
    print("✅ Conversational RAG pipeline is ready.")
    return chain

def generate_with_single_input(prompt: str, 
                              role: str = 'user', 
                              top_p: float = 0.9, 
                              temperature: float = 0.7,
                              max_tokens: int = 500,
                              model: str = "phi3:mini",
                              **kwargs):
    """
    Generates response using local Ollama phi3:mini for a single prompt.
    Returns dict: {'role': 'assistant', 'content': response_text}.
    """
    # Format as chat message
    messages = [{"role": role, "content": prompt}]

    # Generation parameters
    options = {
        "num_predict": max_tokens,
        "temperature": temperature if temperature != 'none' else 0.7,
        "top_p": top_p if top_p != 'none' else 0.9,
        **kwargs
    }

    try:
        response = ollama.chat(model=model, messages=messages, options=options)
        content = response['message']['content']
        return {'role': 'assistant', 'content': content}
    except Exception as e:
        raise Exception(f"Failed to generate response: {e}")

def generate_with_multiple_input(messages: List[Dict], 
                                top_p: float = 0.9, 
                                temperature: float = 0.7,
                                max_tokens: int = 500,
                                model: str = "phi3:mini",
                                **kwargs):
    """
    Generates response using local Ollama phi3:mini for a list of messages.
    Returns dict: {'role': 'assistant', 'content': response_text}.
    """
    # Generation parameters
    options = {
        "num_predict": max_tokens,
        "temperature": temperature if temperature != 'none' else 0.7,
        "top_p": top_p if top_p != 'none' else 0.9,
        **kwargs
    }

    try:
        response = ollama.chat(model=model, messages=messages, options=options)
        content = response['message']['content']
        return {'role': 'assistant', 'content': content}
    except Exception as e:
        raise Exception(f"Failed to generate response: {e}")
  