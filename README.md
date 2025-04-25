# AI-Powered Document Assistant for Subsurface Geoscience Data  
# Document GraphRAG Q&A Assistant

## Overview

This project implements an advanced Question-Answering system using a hybrid Retrieval-Augmented Generation (RAG) approach that leverages both Knowledge Graphs and Vector Embeddings (GraphRAG). It allows users to upload documents (PDF, TXT, Images), processes them to extract structured knowledge into a Neo4j graph database and semantic information into a ChromaDB vector store, and then answers natural language questions based on the combined context, providing source attribution for its answers.

The primary goal is to unlock information typically siloed within documents and make it accessible through an interactive, conversational interface built with Streamlit.

## Features

* **Multi-Format Document Ingestion:** Supports uploading `.pdf`, `.txt`, `.png`, `.jpg`, `.jpeg` files.
* **OCR Integration:** Uses the Mistral OCR API to extract text from PDF and image files.
* **Intelligent Text Chunking:** Employs LangChain's `RecursiveCharacterTextSplitter` to break down documents into meaningful, context-aware chunks.
* **Knowledge Graph Construction:**
    * Utilizes Large Language Models (LLMs) for extracting entities, relationships, and attributes from text chunks.
    * Supports configurable LLMs (local Ollama models like Gemma, Phi3, Llama3 or cloud APIs like Google Gemini, OpenAI GPT).
    * Stores the structured knowledge in a **Neo4j** graph database, modeling attributes as node properties for efficiency.
* **Vector Embeddings:**
    * Generates semantic embeddings for text chunks using SentenceTransformers.
    * Stores embeddings and associated text/metadata in a persistent **ChromaDB** vector store.
* **Hybrid Q&A (GraphRAG):**
    * Generates Cypher queries using an LLM to query the Neo4j graph based on the user's question.
    * Includes a fallback mechanism to perform broader graph searches if the initial specific query fails.
    * Performs semantic similarity search on the ChromaDB vector store.
    * Combines context from both graph and vector retrieval.
    * Synthesizes answers using an LLM based on the retrieved context.
* **Source Attribution:** Displays the specific text chunks retrieved from the vector store that were used to generate the answer, enhancing transparency and trust.
* **Interactive UI:** Built with **Streamlit**, providing:
    * File upload and processing controls.
    * A chat interface for Q&A.
    * Persistent chat history (saved locally to `chat_history.json`).
    * Tabs for viewing processed PDFs and an interactive graph visualization (using `vis.js`).
    * System status indicators.
* **Configuration:** Flexible setup via `config.toml` and environment variables for API keys, database connections, model selection, and processing parameters.
* **Caching:** Optional caching of extracted triples to speed up reprocessing of identical files.

## Architecture Flow

1.  **Upload:** User uploads document(s) via the Streamlit UI.
2.  **Preprocess:**
    * If PDF/Image: Text is extracted using Mistral OCR API.
    * Text is chunked using `RecursiveCharacterTextSplitter`.
3.  **Knowledge Extraction (Parallel):**
    * **Graph Path:** Chunks are sent to an LLM (configured for triple extraction) to generate S-P-O triples and attributes. The `Neo4jExporter` processes these, creating/updating nodes and relationships (or setting properties for attributes) in the Neo4j database.
    * **Vector Path:** Chunks are embedded using a SentenceTransformer model. The embeddings, text, and metadata are stored in the ChromaDB collection.
4.  **Q&A:**
    * User asks a question.
    * **Graph Query:** An LLM attempts to generate a Cypher query based on the question and graph schema.
    * **Neo4j Search:** The Cypher query is executed. If it fails or returns no results, a fallback broad query is attempted based on extracted entities.
    * **Vector Search:** The question is embedded, and ChromaDB is queried for similar text chunks.
    * **Context Combination:** Results from the graph (if any) and vector search are formatted into a combined context.
    * **Answer Synthesis:** The combined context and original question are sent to the Q&A LLM to generate the final answer.
    * **Source Display:** The answer and the retrieved vector source chunks are displayed in the UI.

## Technology Stack

* **Backend:** Python 3.9+
* **Frontend:** Streamlit
* **Knowledge Graph:** Neo4j (via `neo4j` Python driver)
* **Vector Database:** ChromaDB (via `chromadb` client)
* **LLM Interaction:** Custom logic (`llm.py`) using `requests` (supports OpenAI, Gemini, Mistral, Ollama API formats)
* **Text Processing:** `langchain-text-splitters`
* **Embeddings:** `sentence-transformers`
* **OCR:** Mistral AI API (via `mistralai` library)
* **Graph Visualization:** `vis.js` (via `streamlit-agraph` or custom HTML component)
* **Configuration:** `tomli` (or `tomllib` for Python 3.11+), `configparser`
* **Other:** `Pillow` (for image handling if needed), `requests`

## Setup & Installation

1.  **Prerequisites:**
    * Python 3.9 or higher.
    * `pip` for package installation.
    * Access to a Neo4j instance (local, Docker, or Aura).
    * (Optional) Ollama installed and running if using local LLMs.
    * (Optional) API keys for desired cloud services (OpenAI, Google AI Studio, Mistral AI).

2.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate (Linux/macOS)
    source .venv/bin/activate
    # Activate (Windows)
    .\.venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file listing all necessary packages like `streamlit`, `neo4j`, `chromadb`, `sentence-transformers`, `langchain-text-splitters`, `requests`, `tomli`, `mistralai`, `pillow`, `pypdf` (if handling PDFs directly), etc.)*

5.  **Download NLTK Data:** Run this Python script once to download necessary tokenizer data:
    ```python
    import nltk
    nltk.download('punkt')
    ```

6.  **Configure Neo4j:** Ensure your Neo4j instance is running and accessible. Update connection details in `graph_config.ini` or environment variables.

7.  **Configure APIs & Models:**
    * Copy `config.example.toml` to `config.toml`.
    * Edit `config.toml` to add your API keys (OpenAI, Google Gemini, Mistral), select the desired LLM models for Q&A (`[llm.qna]`) and triple extraction (`[llm.triple_extraction]`), and set the base URLs. **It is highly recommended to use environment variables for API keys instead of hardcoding them.**
    * Configure chunking parameters (`chunk_size_chars`, `overlap_chars`) under `[chunking]`.
    * Set ChromaDB path and collection name under `[vector_db]`.
    * Set Neo4j details under `[neo4j]` (can also be in `graph_config.ini` or environment variables).

## Running the Application

1.  Ensure your Neo4j database is running.
2.  If using local Ollama models, ensure the Ollama service is running and the desired models are pulled (`ollama pull <model_name>`).
3.  Activate your virtual environment (if created).
4.  Navigate to the project directory in your terminal.
5.  Run the Streamlit app:
    ```bash
    streamlit run graphrag_app.py
    ```
6.  Open the URL provided by Streamlit in your web browser.
7.  Use the sidebar to upload documents and click "Process Uploaded Document(s)".
8.  Once processing is complete, use the chat interface to ask questions about your documents.

## Core Modules

* `graphrag_app.py`: Main Streamlit application logic, UI definition, and pipeline orchestration.
* `graph_rag_qa.py`: Contains the `GraphRAGQA` class managing the RAG process (query generation, graph/vector retrieval, answer synthesis).
* `neo4j_exporter.py`: Handles storing extracted triples and attributes into the Neo4j graph database.
* `llm.py`: Provides the `call_llm` function for interacting with various LLM APIs (Cloud & Local) and `extract_json_from_text` for parsing responses.
* `text_utils.py`: Contains the `chunk_text` function using LangChain for recursive character splitting.
* `visualization.py`: (Assumed) Contains logic for generating the `vis.js` HTML graph visualization.
* `prompts.py`: (Assumed) Stores the system and user prompts used for LLM interactions (like `MAIN_USER_PROMPT`, `TEXT_TO_CYPHER_SYSTEM_PROMPT`).

## Future Enhancements

* Implement more sophisticated entity extraction for the fallback graph query.
* Add support for more document types.
* Explore different embedding models or fine-tuning.
* Implement more advanced graph algorithms for deeper insights.
* Improve UI/UX, potentially adding asynchronous processing.
* Add options for different graph visualization layouts.



## Demo Video

[https://youtu.be/quiAFph86O8](https://youtu.be/DKbmegaOfbY)

---

## Credits

Built by [Mivaa](https://deepdatawithmivaa.com)  
For the Subsurface Geoscience Community  
Empowering Data-Driven Upstream Workflows

---

## License

MIT License

---
