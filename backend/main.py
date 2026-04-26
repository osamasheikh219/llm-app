from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import requests
import os

# ── Config ────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = "deepseek-ai/DeepSeek-R1"
HF_URL   = "https://router.huggingface.co/v1/chat/completions"


DB_PATH      = "prompts.db"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="LLM Prompt App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# ── Database setup ────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            title   TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    # Seed data (only if empty)
    cur.execute("SELECT COUNT(*) FROM prompts")
    if cur.fetchone()[0] == 0:
        seed = [
            ("Explain a concept simply",
             "Explain the following concept in simple terms that a 10-year-old could understand: Artificial Intelligence"),
            ("Write a short story",
             "Write a short, imaginative 3-sentence story set in a futuristic city on Mars."),
            ("Give career advice",
             "I am a computer science student. Give me 3 practical tips to land my first DevOps job."),
            ("Summarize a topic",
             "Provide a concise 5-sentence summary of how the internet works, from typing a URL to seeing a webpage."),
            ("Motivational quote + explanation",
             "Give me an original motivational quote about learning and then explain it in 2 sentences."),
        ]
        cur.executemany("INSERT INTO prompts (title, content) VALUES (?, ?)", seed)
    conn.commit()
    conn.close()

init_db()

# ── Models ────────────────────────────────────────────────────────────────────
class SubmitRequest(BaseModel):
    prompt_id: int

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse("../frontend/index.html")

@app.get("/api/prompts")
def list_prompts():
    """Return all prompts (id + title) for the dropdown."""
    conn = get_db()
    rows = conn.execute("SELECT id, title FROM prompts ORDER BY id").fetchall()
    conn.close()
    return [{"id": r["id"], "title": r["title"]} for r in rows]

@app.post("/api/submit")
def submit_prompt(body: SubmitRequest):
    """Fetch prompt from DB → send to LLM → return response."""
    # 1. Query database
    conn = get_db()
    row  = conn.execute("SELECT * FROM prompts WHERE id = ?", (body.prompt_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Prompt not found")

    prompt_text = row["content"]

    # 2. Call HuggingFace Inference API
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
    }

    try:
        hf_resp = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        hf_resp.raise_for_status()
        result = hf_resp.json()
        llm_output = result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="LLM API timed out. Try again.")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Return to frontend
    return {
        "prompt_id":    row["id"],
        "prompt_title": row["title"],
        "prompt_text":  prompt_text,
        "llm_response": llm_output,
    }
