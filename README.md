# Paper Embedding and RAG Pipeline

This project demonstrates a full pipeline for building a Retrieval-Augmented Generation (RAG) system for scientific papers. It fetches research papers, processes them into a searchable vector database, and provides a retrieval script to query the database.

The pipeline consists of the following main components:

1.  **Paper Fetcher**: Downloads scientific papers from arXiv using the Semantic Scholar API to get metadata and citation information.
2.  **PDF to Markdown Converter**: Uses the `marker-pdf` library to convert the downloaded PDFs into clean Markdown files, preserving structure and text.
3.  **Embedding Pipeline**:
    *   **Text Chunking**: Splits the Markdown files into smaller, manageable chunks.
    *   **Contextual Enrichment**: Uses a Large Language Model (Qwen-2.5-7B-Instruct) to generate a concise, one-sentence summary for each chunk to improve retrieval context.
    *   **Citation Resolution**: Parses citation markers (e.g., `[1]`) and maps them to their corresponding Semantic Scholar paper IDs.
    *   **Embedding Generation**: Creates vector embeddings for each enriched chunk using the `nomic-ai/nomic-embed-text-v1.5` model.
    *   **Vector Database Storage**: Stores the original chunks, their embeddings, and associated metadata (source paper, enriched context, citation IDs) in a ChromaDB vector database.
4.  **Retriever**: A script that takes a user query, embeds it, and retrieves the most relevant text chunks from the vector database.

## Project Structure

```
.
├─── seed_papers/           # PDFs of the seed papers are downloaded here
├─── seed_papers_md/        # Markdown versions of the papers are created here
├─── paper_rag_db/          # The ChromaDB vector database
├─── paper_fetcher.py       # Fetches papers and their citations
├─── embeddings.py          # Processes papers and creates embeddings
├─── retriever.py           # Queries the vector database
├─── runner.ipynb           # Jupyter notebook to run the full pipeline
├─── citation_resolver_map.json # Maps seed papers to their citations
└─── README.md
```

## How to Run

### 1. Prerequisites

You need Python 3.8+ and a GPU with at least 8GB of VRAM is recommended for the OCR and embedding steps. If you don't have a GPU, the process will run on the CPU but will be significantly slower.

### 2. Installation

The required Python packages are listed in the `runner.ipynb` notebook. You can install them using pip:

```bash
pip install chromadb bitsandbytes langchain-text-splitters thefuzz marker-pdf sentence-transformers transformers torch
```

Optionally, if you want to use the Semantic Scholar API with an API key for higher rate limits, create a file named `api_key.json` in the root of the project with the following format:

```json
{
  "x-api-key": "YOUR_API_KEY"
}
```

### 3. Running the Pipeline

The easiest way to run the full pipeline is to use the `runner.ipynb` notebook. It contains cells that execute each step of the process in order:

1.  **Fetch Papers**: `fetch_papers()` from `paper_fetcher.py` will download the PDFs of the papers defined in `SEED_PAPERS` and create the `citation_resolver_map.json`.
2.  **Convert PDFs to Markdown**: `ocr_directory_marker()` from `embeddings.py` will process the PDFs in `seed_papers/` and save the Markdown versions in `seed_papers_md/`.
3.  **Create Embeddings**: `run_pipeline()` from `embeddings.py` will process the Markdown files, create embeddings, and store them in the `paper_rag_db` ChromaDB database.
4.  **Query the Database**: The notebook demonstrates how to use the `query_db()` function from `retriever.py` to ask questions and get relevant results from the papers.

You can also run the individual scripts from the command line:

```bash
# To fetch papers
python -c 'from paper_fetcher import fetch_papers; fetch_papers()'

# To process PDFs and create embeddings
python embeddings.py

# To query the database with some example questions
python -c 'from retriever import query_db; query_db(["What is the attention mechanism?", "How does BERT work?"])'
```
