from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from faq_rag import load_faq_from_google_sheet, embed_questions, build_faiss_index, match_user_question, init_db, llm_fallback
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID')

app = FastAPI(title="Hawala RAG API", version="1.0.0")

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    user_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str
    match_score: float
    used_llm: bool
    llm_cost: Optional[float] = None
    matched_faq: str

# Global variables to store initialized components
faq_df = None
index = None
dim = None
embeddings = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global faq_df, index, dim, embeddings
    
    print("Initializing RAG system...")
    init_db()
    
    try:
        # Load FAQ data
        faq_df = load_faq_from_google_sheet(GOOGLE_SHEET_ID)
        print(f"Loaded {len(faq_df)} FAQs.")
        
        # Create embeddings and index
        questions = [str(q) for q in faq_df['Question'].tolist() if q and str(q).strip() != ""]
        print(f"Creating embeddings for {len(questions)} questions...")
        embeddings = embed_questions(questions)
        index, dim = build_faiss_index(embeddings)
        print("RAG system ready!")
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print("Server will start but RAG functionality will be limited.")
        # Set defaults to prevent crashes
        faq_df = None
        index = None
        dim = None
        embeddings = None

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer"""
    global faq_df, index, dim, embeddings
    
    if faq_df is None or index is None:
        # Fallback to LLM only if RAG system not initialized
        print("RAG system not initialized, using LLM fallback only")
        try:
            from faq_rag import llm_fallback
            answer, llm_cost = llm_fallback(request.question)
            return QuestionResponse(
                answer=answer,
                match_score=0.0,
                used_llm=True,
                llm_cost=llm_cost,
                matched_faq="LLM_FALLBACK_ONLY"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
    try:
        answer, score, matched_faq, used_llm, llm_cost = match_user_question(
            request.user_id, 
            request.question, 
            faq_df, 
            index, 
            dim, 
            embeddings
        )
        
        return QuestionResponse(
            answer=answer,
            match_score=score,
            used_llm=used_llm,
            llm_cost=llm_cost,
            matched_faq=matched_faq
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "faqs_loaded": len(faq_df) if faq_df is not None else 0}

@app.get("/stats")
async def get_stats():
    """Get usage statistics"""
    import sqlite3
    conn = sqlite3.connect('faq_logs.db')
    c = conn.cursor()
    
    # Get total queries
    c.execute("SELECT COUNT(*) FROM logs")
    total_queries = c.fetchone()[0]
    
    # Get LLM usage
    c.execute("SELECT COUNT(*) FROM logs WHERE llm_used IS NOT NULL")
    llm_queries = c.fetchone()[0]
    
    # Get total cost
    c.execute("SELECT SUM(llm_cost) FROM logs WHERE llm_cost IS NOT NULL")
    total_cost = c.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "total_queries": total_queries,
        "llm_queries": llm_queries,
        "faq_queries": total_queries - llm_queries,
        "total_cost": round(total_cost, 4),
        "cost_savings": round((total_queries - llm_queries) * 0.002, 4)  # Estimated savings
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 