import os
import sqlite3
import pandas as pd
import openai
import faiss
from dotenv import load_dotenv
from typing import Tuple
from datetime import datetime
import numpy as np

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID')

openai.api_key = OPENAI_API_KEY

# --- Google Sheets Loader ---
def load_faq_from_google_sheet(sheet_id: str) -> pd.DataFrame:
    import gspread
    gc = gspread.service_account(filename='ameenai-465403-d0b8705da954.json')
    sh = gc.open_by_key(sheet_id)
    worksheet = sh.sheet1
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    return df[['Question', 'Answer']]

# --- Embedding Function ---
def embed_questions(questions: list, max_retries=3) -> list:
    all_embeddings = []
    batch_size = 100
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        
        for attempt in range(max_retries):
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                all_embeddings.extend([d['embedding'] for d in response['data']])
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception(f"Failed to embed questions after {max_retries} attempts: {str(e)}")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                
    return all_embeddings

# --- FAISS Indexer ---
def build_faiss_index(embeddings: list) -> Tuple[faiss.IndexFlatIP, int]:
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    xb = np.array(embeddings).astype('float32')
    faiss.normalize_L2(xb)
    index.add(xb)
    return index, dim

# --- SQLite Logger ---
def init_db(db_path='faq_logs.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        question TEXT,
        question_embedding TEXT,
        match_score REAL,
        matched_faq TEXT,
        answer TEXT,
        timestamp TEXT,
        llm_used TEXT,
        llm_cost REAL
    )''')
    conn.commit()
    conn.close()

def log_query(user_id, question, question_embedding, match_score, matched_faq, answer, llm_used=None, llm_cost=None, db_path='faq_logs.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT INTO logs (user_id, question, question_embedding, match_score, matched_faq, answer, timestamp, llm_used, llm_cost)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_id, question, question_embedding, match_score, matched_faq, answer, datetime.utcnow().isoformat(), llm_used, llm_cost))
    conn.commit()
    conn.close()

def get_logged_questions_and_answers(db_path='faq_logs.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT question, answer, question_embedding FROM logs WHERE answer IS NOT NULL AND question_embedding IS NOT NULL')
    rows = c.fetchall()
    conn.close()
    questions = [row[0] for row in rows]
    answers = [row[1] for row in rows]
    embeddings = [eval(row[2]) for row in rows]  # Convert string back to list
    return questions, answers, embeddings

def embed_single_question(question: str, max_retries=3) -> list:
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                input=[question],
                model="text-embedding-3-small"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Single embedding attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(f"Failed to embed single question after {max_retries} attempts: {str(e)}")
            import time
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

# --- LLM Fallback Function ---
def llm_fallback(user_question: str) -> (str, float):
    system_prompt = """
You are a financial assistant for Hawala, a remittance and financial planning platform.

Hawala's capabilities:
- Money transfers and remittances
- Financial planning and budgeting tools
- Project-based savings goals (users can create a project/goal in the financial planner tool tab)
- Virtual bank accounts
- Integration with local banking systems

IMPORTANT: When users ask about saving for a goal (like Umrah, marriage, etc.), ALWAYS tell them you can create a project/goal for them in the financial planner tool tab. Do not just recommend it—explicitly offer: 'Would you like me to create this project/goal for you now?'

When users ask about money transfers, explain our remittance services.

Always be helpful and guide users toward using Hawala's specific features.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        max_tokens=256,
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()
    total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 256
    cost = (total_tokens / 1000) * 0.002
    return answer, cost

# --- Main Matching Function ---
def match_user_question(user_id: str, user_question: str, faq_df: pd.DataFrame, index, dim, embeddings, threshold=0.8):
    # Embed user question
    user_emb = embed_questions([user_question])[0]
    user_emb_np = np.array(user_emb).astype('float32').reshape(1, -1)
    faiss.normalize_L2(user_emb_np)
    # Search FAQ
    D, I = index.search(user_emb_np, 1)
    score = float(D[0][0])
    idx = int(I[0][0])
    matched_faq = faq_df.iloc[idx]['Question']
    answer = faq_df.iloc[idx]['Answer'] if score > threshold else None
    if answer:
        log_query(user_id, user_question, str(user_emb), score, matched_faq, answer, llm_used=None, llm_cost=None)
        return answer, score, matched_faq, False, None
    # --- Check logs for similar user queries ---
    logged_questions, logged_answers, logged_embeddings = get_logged_questions_and_answers()
    if logged_questions:
        # Use pre-stored embeddings 
        logged_embeddings_np = np.array(logged_embeddings).astype('float32')
        faiss.normalize_L2(logged_embeddings_np)
        # Compute similarity
        D_logs = np.dot(logged_embeddings_np, user_emb_np.T).flatten()
        best_idx = int(np.argmax(D_logs))
        best_score = float(D_logs[best_idx])
        if best_score > threshold:
            log_query(user_id, user_question, str(user_emb), best_score, 'PAST_USER_QUERY', logged_answers[best_idx], llm_used=None, llm_cost=None)
            return logged_answers[best_idx], best_score, 'PAST_USER_QUERY', False, None
    # --- LLM fallback ---
    llm_answer, llm_cost = llm_fallback(user_question)
    log_query(user_id, user_question, str(user_emb), score, matched_faq, llm_answer, llm_used="gpt-3.5-turbo", llm_cost=llm_cost)
    return llm_answer, score, matched_faq, True, llm_cost

# --- CLI Test Function ---
def main():
    print('Initializing...')
    init_db()
    faq_df = load_faq_from_google_sheet(GOOGLE_SHEET_ID)
    print(f'Loaded {len(faq_df)} FAQs.')
    questions = [str(q) for q in faq_df['Question'].tolist() if pd.notnull(q) and str(q).strip() != ""]
    embeddings = embed_questions(questions)
    index, dim = build_faiss_index(embeddings)
    print('Ready for questions!')
    while True:
        user_id = input('User ID: ')
        user_q = input('Ask a question: ')
        answer, score, matched_faq, used_llm, llm_cost = match_user_question(user_id, user_q, faq_df, index, dim, embeddings)
        if not used_llm:
            print(f'FAQ Match (score={score:.2f}): {answer}')
        else:
            print(f'LLM Fallback (score={score:.2f}, cost=${llm_cost:.4f}): {answer}')

if __name__ == '__main__':
    main() 