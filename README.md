# CogniVerse: A Multimodal AI Study Buddy

CogniVerse is a sophisticated, conversational AI study buddy designed to provide grounded, textbook-specific answers for students. Unlike generic chatbots that pull from the entire internet, CogniVerse uses a Retrieval-Augmented Generation (RAG) pipeline to ensure every answer is derived exclusively from a user-provided textbook, including its text, tables, and diagrams.

This project is a deep dive into building a robust, multimodal RAG system from the ground up, tackling real-world challenges like data quality, retrieval relevance, and model obedience.

## The Problem

Studying for university exams requires precise, context-specific information. Generic AI assistants often "hallucinate" or provide answers that, while correct in a general sense, do not match the specific terminology, structure, or examples from the required course material. This makes them unreliable for exam preparation.

CogniVerse was built to solve this problem by creating a "closed-loop" AI that can only access and reason over a single source of truth: the student's own textbook.

## Features

* **Conversational Interface**: A sleek, two-column UI with a chat on the left and retrieved context on the right.
* **Grounded Answers**: The AI is strictly forbidden from using outside knowledge, preventing hallucinations.
* **Multimodal Understanding**: Ingests and analyzes not just text, but also tables and complex diagrams from the source material.
* **Source Transparency**: For every answer, the user can see the exact text chunks and images the AI used as context.
* **Advanced RAG Pipeline**: Utilizes state-of-the-art techniques like HyDE and Re-ranking to deliver highly relevant and accurate results.

## Technical Architecture

CogniVerse is built on a modern RAG architecture that emphasizes data quality and retrieval accuracy above all else.

#### Tech Stack

* **Orchestration Framework**: [LangChain](https://www.langchain.com/)
* **Backend**: Python with [Flask](https://flask.palletsprojects.com/)
* **Frontend**: [Next.js](https://nextjs.org/) (React Framework) with [shadcn/ui](https://ui.shadcn.com/)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
* **Document Store**: LangChain `LocalFileStore`
* **LLMs & APIs**:
    * **Google Gemini 1.5 Flash**: Used for high-quality summarization during data processing and for the final, context-obedient answer generation.
    * **Cohere**: Used for its powerful Re-ranking model to improve retrieval relevance.
    * **Ollama (`phi3:mini`)**: Used locally for the fast and efficient HyDE query transformation step.
* **Embedding Model**: `BAAI/bge-large-en-v1.5` from [Hugging Face](https://huggingface.co/) for creating vector representations of the summaries.

#### Architectural Flow

1.  **Data Processing (One-Time Setup)**:
    * **Manual Curation**: To ensure perfect data quality, text is manually extracted into a Markdown file, and images are captured via screenshots. This bypasses the unreliability of automated PDF parsers on complex layouts.
    * **AI-Powered Summarization**: The clean text and image data is processed by the **Google Gemini 1.5 Flash API** in batches to generate high-quality, semantically rich summaries for every chunk and image.
    * **Multi-Vector Storage**: The summaries are embedded and stored in a **ChromaDB** vector store. The original, full-size text and image paths are stored in a simple `LocalFileStore`.

2.  **Live RAG Chain (Per Query)**:
    * **Query Transformation (HyDE)**: The user's question is first sent to a fast, local LLM (`phi3:mini`) to generate a "hypothetical document." This transforms the query into a statement, which dramatically improves retrieval accuracy.
    * **Multimodal Retrieval**: The hypothetical answer is used to search the ChromaDB vector store, retrieving the most relevant summaries. The system then fetches the corresponding full-size documents (text and images) from the `LocalFileStore`.
    * **Intelligent Re-ranking**: The initial results are passed to a **Cohere Re-ranker**, which intelligently filters and prioritizes the top 3 most relevant documents, discarding any with a weak signal.
    * **Obedient, Synthesized Answer**: The final, rich context is sent to the **Gemini 1.5 Flash API** to generate a single, cohesive, tutor-like explanation that can analyze both the text and the diagrams. The use of a highly-aligned API model for this final step guarantees obedience to the provided context.

## Project Structure

CogniVerse/
├── data/                  # Source data (e.g., textbook.md, manual_images/)
├── doc_store/             # Persistent key-value store for full documents
├── frontend/              # Next.js frontend application
├── processed_data/        # Pickled lists of processed text, tables, images
├── src/
│   ├── backend.py         # The final, working Flask backend for the RAG chain
│   ├── data_processor.py  # Script for the one-time data processing and indexing
│   └── ...                # Other scripts for debugging and utilities
├── vector_store_chroma/   # Persistent ChromaDB vector store
├── .env                   # Environment variables (API keys)
└── requirements.txt       # Python dependencies

## Getting Started

### Prerequisites

* Python 3.9+
* Node.js and npm/yarn
* An Ollama server running with the `phi3:mini` model pulled (`ollama pull phi3:mini`)
* API keys for Google (Gemini) and Cohere

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/CogniVerse.git](https://github.com/your-username/CogniVerse.git)
    cd CogniVerse
    ```

2.  **Backend Setup:**
    * Create a Python virtual environment and activate it.
    * Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    * Create a `.env` file in the root directory and add your API keys:
        ```
        GOOGLE_API_KEY="your_google_api_key"
        COHERE_API_KEY="your_cohere_api_key"
        ```

3.  **Frontend Setup:**
    * Navigate to the frontend directory:
        ```bash
        cd frontend
        ```
    * Install Node.js dependencies:
        ```bash
        npm install
        ```

### Running the Application

1.  **Prepare Your Data:**
    * Place your textbook content in `data/textbook.md`.
    * Place your manually captured images in `data/manual_images/`.
    * Run the data processor script to create the vector stores. This is a one-time process.
        ```bash
        python src/data_processor.py
        ```

2.  **Start the Backend:**
    * Make sure your Ollama server is running.
    * Run the Flask application from the root directory:
        ```bash
        python src/backend.py
        ```
    The backend will be running at `http://127.0.0.1:5000`.

3.  **Start the Frontend:**
    * In a new terminal, navigate to the `frontend` directory and run:
        ```bash
        npm run dev
        ```
    The application will be accessible at `http://localhost:3000`.

## Acknowledgements

This project was built using the powerful LangChain library and relies on a number of incredible open-source and API-based tools, including:

* Google Gemini
* Cohere
* Ollama
* ChromaDB & FAISS
* Hugging Face Transformers
* Next.js & shadcn/ui