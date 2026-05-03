from fastapi import FastAPI, HTTPException, UploadFile, File
import boto3
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import requests
import os
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL     = "deepseek-ai/DeepSeek-R1"
HF_URL       = "https://router.huggingface.co/v1/chat/completions"
DB_PATH      = "prompts.db"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Prompt App",
    description="Multi-tier AI application with FastAPI + SQLite + HuggingFace",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

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

class FreeTextRequest(BaseModel):
    prompt_text: str

# ── Helper: call HuggingFace ──────────────────────────────────────────────────
def call_llm(prompt_text: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    try:
        hf_resp = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        hf_resp.raise_for_status()
        result = hf_resp.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="LLM API timed out. Try again.")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serve the frontend HTML page."""
    return FileResponse("../frontend/index.html")

@app.get("/api/prompts", tags=["Prompts"])
def list_prompts():
    """Return all available prompts (id + title) for the dropdown."""
    conn = get_db()
    rows = conn.execute("SELECT id, title FROM prompts ORDER BY id").fetchall()
    conn.close()
    return [{"id": r["id"], "title": r["title"]} for r in rows]

@app.post("/api/submit", tags=["Prompts"])
def submit_prompt(body: SubmitRequest):
    """
    Accept a prompt ID → fetch from DB → call LLM → return response.
    Used by the frontend UI.
    """
    conn = get_db()
    row  = conn.execute("SELECT * FROM prompts WHERE id = ?", (body.prompt_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Prompt not found")

    llm_output = call_llm(row["content"])

    return {
        "prompt_id":    row["id"],
        "prompt_title": row["title"],
        "prompt_text":  row["content"],
        "llm_response": llm_output,
    }

@app.post("/api/submit-text", tags=["Lambda"])
def submit_free_text(body: FreeTextRequest):
    """
    Accept any raw prompt text → call LLM → return response.
    Used by AWS Lambda for event-driven S3 processing.
    """
    if not body.prompt_text.strip():
        raise HTTPException(status_code=400, detail="prompt_text cannot be empty")

    llm_output = call_llm(body.prompt_text)

    return {
        "prompt_text":  body.prompt_text,
        "llm_response": llm_output,
    }

@app.get("/api/health", tags=["System"])
def health_check():
    """Check if the API is running correctly."""
    return {
        "status":    "ok",
        "model":     HF_MODEL,
        "token_set": bool(HF_API_TOKEN),
    }

S3_BUCKET = "llm-app-prompts-osama"

@app.post("/api/upload-prompt", tags=["S3"])
async def upload_prompt(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files allowed")
    s3 = boto3.client("s3")
    key = f"inputs/{file.filename}"
    content = await file.read()
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content)
    return {"uploaded": key}

@app.get("/api/outputs", tags=["S3"])
def list_outputs():
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="outputs/")
    files = []
    for obj in response.get("Contents", []):
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": obj["Key"]},
            ExpiresIn=3600
        )
        files.append({
            "key": obj["Key"],
            "filename": obj["Key"].split("/")[-1],
            "size": obj["Size"],
            "last_modified": obj["LastModified"].isoformat(),
            "download_url": url
        })
    return sorted(files, key=lambda x: x["last_modified"], reverse=True)
