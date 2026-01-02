This project is a **RAG (Retrieval-Augmented Generation)** application built with **FastAPI**, designed to allow the ingestion of PDF documents and asking questions about their content using Large Language Models (LLMs).

The system processes documents, generates vector embeddings, and stores them in a **PostgreSQL** database with the **pgvector** extension. Queries can be answered using **Google Gemini** or local models via **Ollama** (such as Llama 3).

## Features

*   **Document Upload**: Accepts PDF files, extracts text, and splits it into chunks.
*   **Semantic Search**: Uses embeddings (via `sentence-transformers`) to find relevant sections in the documents.
*   **LLM Integration**:
    *   **Google Gemini**: For fast, high-quality cloud-based responses.
    *   **Ollama**: For running local models (e.g., Llama 3).
*   **REST API**: Simple interface built with FastAPI for upload and queries.
*   **Web Interface:** There's a simple web interface built to communicate with the system effortlessly.

## Demo

You can have a preview without download anything in the following [notebook](https://colab.research.google.com/drive/13Fen8ZioJPHrGCFYvW5CSztQWw2TRoTC?usp=sharing)!

## Prerequisites

*   **Docker** and **Docker Compose**.
*   **Python 3.10+**.
*   **Ollama** (optional, if you wish to use local models).
*   **Google API Key** (optional, if you wish to use Gemini).

## Installation and Execution

### 1. Configure the Database

The project uses Docker Compose to spin up a PostgreSQL instance with the `pgvector` extension enabled.

```bash
docker-compose up -d
```

### 2. Configure the Python Environment

It is recommended to use a virtual environment:

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root following `.env.example` template:

### 4. (Optional) Configure Ollama 

If you plan to use the local models, ensure you have Ollama installed and running locally. The project is configured to use the `llama3` model by default.

```bash
ollama pull llama3
ollama serve
```

### 5. Run the Application

Run the FastAPI server:

```bash
fastapi dev src/app.py
```

The API will be available at `http://localhost:8000/`.

## Web Interface
```bash
cd web/
python3 -m http.server 8002
# or use an extension like live server, it will work the same.
```

## How to Use

You can interact with the API through the automatic documentation (Swagger UI) at `http://localhost:8000/docs`.

1.  **PDF Upload**:
    *   Use the `POST /upload/` endpoint to upload a file path.
    *   Use the `POST /upload-file/` endpoint to upload a PDF file.

2.  **Ask Questions**:
    *   Use `POST /query/` with `{"question": "Your question here"}`.

3. **Changing Models**
    *   Change the .env `LLM_MODEL` variable to change the model in use. Available now: [`gemini`, `ollama`]

3.  **Clear Database**:
    *   Use `DELETE /clear-database/` to remove all documents and embeddings.
