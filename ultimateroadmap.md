 System Architecture
                    User
                      │
              Streamlit Dashboard
                      │
         ┌────────────┴────────────┐
         │                         │
    Upload Documents         Chat Interface
         │                         │
         └────────────┬────────────┘
                      │
              Document Pipeline
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   PDF Loader     DOCX Loader    TXT Loader
        │             │             │
        └─────────────┴─────────────┘
                      │
               Text Cleaning
                      │
                Smart Chunking
                      │
          SentenceTransformer
                      │
               ChromaDB Storage
                      │
         Semantic Search + BM25
                      │
                 Reranker
                      │
            Context Builder
                      │
              Local LLM (Ollama)
                      │
              Final Response
                      │
        Sources + Confidence Score

Folder Structure

knowledgeforge-ai/
│
├── app/
│   ├── main.py
│   ├── ui.py
│   ├── chat.py
│   └── settings.py
│
├── rag/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── prompts.py
│   └── generator.py
│
├── loaders/
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   ├── txt_loader.py
│   └── markdown_loader.py
│
├── models/
│   ├── embedding.py
│   └── llm.py
│
├── utils/
│   ├── logger.py
│   ├── config.py
│   └── helpers.py
│
├── database/
│   └── chroma/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE

Tech Stack

| Layer            | Tool                          |
| ---------------- | ----------------------------- |
| Language         | Python                        |
| UI               | Streamlit                     |
| Backend          | FastAPI (optional API layer)  |
| Embeddings       | Sentence Transformers         |
| Vector Database  | ChromaDB                      |
| Hybrid Search    | BM25 + Dense Retrieval        |
| Reranking        | CrossEncoder (`bge-reranker`) |
| Local LLM        | Ollama (Qwen, Llama, Gemma)   |
| Document Parsing | PyMuPDF, python-docx          |
| OCR              | EasyOCR or Tesseract          |
| Logging          | Loguru                        |
| Testing          | pytest                        |
| Packaging        | Docker                        |


Pipeline
1. Document Ingestion

Support:

PDFs
DOCX
TXT
Markdown
CSV
HTML
Python files

Extract metadata:

Filename
Page number
Creation date
Document type

2. Text Cleaning
Remove duplicate whitespace
Normalize Unicode
Remove headers/footers (when possible)
Preserve section titles

3. Smart Chunking

Instead of fixed-size chunks, use semantic or recursive chunking with overlap.

Each chunk stores:

Document ID
Chunk ID
Page
Heading
Content

4. Embeddings
   Use a local embedding model such as all-MiniLM-L6-v2 to convert chunks into vectors.

   5. Vector Database
Store:
Embedding
Metadata
Chunk text

This enables semantic retrieval.


6. Hybrid Retrieval

Combine:

Dense vector search
BM25 keyword search

Merge the results to improve recall.

7. Reranking

Score the retrieved chunks with a cross-encoder so the most relevant context is passed to the LLM.

8. Context Builder

Assemble the top-ranked chunks while respecting the model's context window.

9. Local LLM

Run through Ollama (or another local inference engine) and instruct the model to:

Answer only from the provided context
Say "I don't know" if the answer isn't supported
Cite source chunks

10. Response

Display:

Answer
Source documents
Page numbers
Confidence indicator (clearly labeled as an estimate if you derive it)
        