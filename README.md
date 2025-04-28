# Local PDF RAG - Embedding and Retrieval Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Embeddings](https://img.shields.io/badge/Embeddings-BAAI/bge--small--en--v1.5-yellow)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Vector Store](https://img.shields.io/badge/Vector_Store-Chroma_DB-red)](https://www.trychroma.com/)
[![PDF Processing](https://img.shields.io/badge/PDF_Processing-pdfplumber-orange)](https://github.com/jsvine/pdfplumber)
[![Chunking](https://img.shields.io/badge/Chunking-LangChain-blue)](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)

This project implements a local pipeline to process PDF documents, generate embeddings using the `BAAI/bge-small-en-v1.5` model from Hugging Face, store them in a persistent Chroma DB instance, and allow interactive querying to retrieve relevant text chunks.

**Note:** This script focuses on the **Embedding** and **Retrieval** stages. It does **not** currently integrate a Large Language Model (LLM) to generate answers based on the retrieved context. It shows you the relevant source text passages directly.

## Overview

The `main.py` script performs the following steps:

1.  **Scans a Directory:** Looks for PDF files in a specified local directory.
2.  **Incremental Processing:** Checks a `data.json` file to avoid reprocessing files that have already been processed.
3.  **PDF Text Extraction:** Uses `pdfplumber` to extract text content page by page from new PDF files.
4.  **Data Persistence:** Saves the extracted raw text and metadata (filename, page) to `data.json`.
5.  **Text Chunking:** Uses `langchain_text_splitters.RecursiveCharacterTextSplitter` to break down the extracted text into smaller, overlapping chunks. Saves these chunks to `chunks.json`.
6.  **Embedding Generation:** Loads the `BAAI/bge-small-en-v1.5` Sentence Transformer model and tokenizer using the `transformers` library. Generates embeddings for each text chunk. Saves chunks with their embeddings to `embedded_chunk.json`.
7.  **Vector Storage:** Initializes a persistent Chroma DB client locally. Adds the text chunks, their corresponding embeddings, and metadata (source document/page) to a Chroma DB collection named `info`.
8.  **Interactive Retrieval:** Enters a loop where the user can input questions. The script embeds the question using the same BGE model and queries the Chroma DB collection to find the most semantically similar text chunks.
9.  **Display Results:** Prints the content and source (filename and page) of the top 3 retrieved chunks.

## Features

*   **Local First:** Operates entirely locally after the initial embedding model download.
*   **PDF Focused:** Directly processes `.pdf` files using `pdfplumber`.
*   **Specific Embedding Model:** Utilizes the efficient `BAAI/bge-small-en-v1.5` model via Hugging Face `transformers`.
*   **Persistent Vector Store:** Uses Chroma DB for durable local storage of embeddings.
*   **Incremental Updates:** Only processes new PDF files added to the data directory.
*   **Source Tracking:** Retrieved chunks include metadata indicating the original PDF file and page number.
*   **Interactive Querying:** Allows users to ask questions in a simple command-line loop.

## Prerequisites

*   **Python:** Version 3.8 or higher recommended.
*   Required Python libraries installed (see imports in `main.py`: `pdfplumber`, `transformers`, `torch`, `chromadb`, `langchain-text-splitters`, etc.).
*   **Data Directory:** A folder containing the PDF files to be processed.
*   **Writable Path:** A location where the Chroma database can be saved persistently.
*   **Internet Connection:** Required only for the *first run* to download the `BAAI/bge-small-en-v1.5` model from Hugging Face.

## Usage

1.  **Prepare Environment:** Ensure Python and necessary libraries are installed. Modify the `directory` variable and the `chromadb.PersistentClient` path within `main.py` to point to your PDF folder and desired database storage location, respectively. Place your PDF files in the specified data directory.
2.  **Run the Script:** Execute `main.py` from your terminal within the project directory.
    ```bash
    python main.py
    ```
3.  **Initial Processing:** On the first run (or when new PDFs are added), the script will process the documents, create embeddings, and add them to Chroma DB. This may take time depending on the number and size of PDFs and your hardware. You'll see output indicating progress. Intermediate JSON files (`data.json`, `chunks.json`, `embedded_chunk.json`) will also be created/updated.
4.  **Ask Questions:** Once the processing is complete, you will be prompted:
    ```
    Ask a question (or type 'exit' to quit):
    ```
    Enter your question about the content of your PDFs and press Enter.
5.  **View Results:** The script will print the top 3 most relevant text chunks found in your documents, along with their source file and page number.
6.  **Exit:** Type `exit` and press Enter to stop the script.

## Future Work

*   **LLM Integration:** Add a component to feed the retrieved context and the user query to a local LLM (e.g., using Ollama, llama.cpp, or directly via `transformers`) to generate a concise natural language answer.
*   **Configuration File:** Move hardcoded paths (`directory`, ChromaDB `path`) to a configuration file (e.g., `config.yaml`, `.env`) for easier setup.
*   **Error Handling:** Add more robust error handling (e.g., for corrupted PDFs, file access issues).
*   **Support More File Types:** Extend document loading to handle `.txt`, `.docx`, etc.
*   **Command-Line Arguments:** Add arguments for query, collection name, number of results (`n_results`), etc., instead of relying solely on the interactive loop for queries.

## Contributing

Contributions or suggestions are welcome! Feel free to open an issue or submit a pull request if you have ideas for improvements.

## License

This project is licensed under the MIT License.