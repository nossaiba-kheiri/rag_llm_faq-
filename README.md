# Hawala FAQ RAG Backend

This project is a Retrieval-Augmented Generation (RAG) backend for answering FAQ and financial questions using OpenAI and Google Sheets.

## Features

- **Google Sheets FAQ Loader:** Loads FAQ questions and answers from a Google Sheet for easy updates.
- **OpenAI Embeddings:** Uses OpenAI's embedding API to represent questions as vectors for semantic search.
- **FAISS Vector Search:** Finds the most similar FAQ to a user's question using fast vector similarity search.
- **User Query Logging:** Logs all user questions, answers, and embeddings in a SQLite database for analytics and learning.
- **Embedding Comparison:**
  - When a user asks a question, the system first compares its embedding to all FAQ embeddings.
  - If no FAQ is a good match, it compares to all previous user questions (using stored embeddings) to see if a similar question was already answered.
  - Only if no good match is found, the system falls back to the LLM (OpenAI GPT) for a fresh answer.
- **LLM Fallback:** Uses OpenAI GPT to answer questions not covered by the FAQ or past user queries.
- **Cost Tracking:** Tracks LLM usage and cost per query.
- **REST API:** FastAPI server with endpoints for asking questions, health check, and usage stats.

## Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd ameenai
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_SHEET_ID=your_google_sheet_id_here
```

### 5. Run the backend (choose one of the following):

**Option A: Run the API server (recommended for integration with frontend or external tools)**
```bash
python api_server.py
```
The API will be available at `http://localhost:8000`.

**Option B: Run the CLI script (for quick manual testing in the terminal)**
```bash
python faq_rag.py
```
This will start an interactive prompt where you can enter user IDs and questions directly.

### 6. Test the API
- Health check: `GET http://localhost:8000/health`
- Ask a question: `POST http://localhost:8000/ask` with JSON body:
  ```json
  {
    "user_id": "test_user",
    "question": "test_question"
  }
  ```

