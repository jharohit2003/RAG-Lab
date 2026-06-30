"""
main.py  --  The web server.

Each RAG step gets its OWN endpoint. That is deliberate: the frontend calls them
one at a time so you can watch the pipeline advance stage by stage. A production
app would usually collapse these into one call, but for learning, separation is
the whole idea.

Run with:   uvicorn main:app --reload --app-dir backend
Then open:  http://localhost:8000
"""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_pipeline as rag

app = FastAPI(title="RAG Learning Lab")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ----- request bodies ------------------------------------------------------- #
class LoadTextBody(BaseModel):
    text: str
    source_name: str = "pasted-text"


class ChunkBody(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 80


class RetrieveBody(BaseModel):
    question: str
    top_k: int = 4


class GenerateBody(BaseModel):
    question: str
    top_k: int = 4


class RunAllBody(BaseModel):
    question: str


# ----- helpers -------------------------------------------------------------- #
def _read_upload(file: UploadFile) -> str:
    """Turn an uploaded .txt, .md, or .pdf into plain text (step 1)."""
    raw = file.file.read()
    name = (file.filename or "").lower()

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(500, "pypdf not installed; run pip install pypdf")
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    # Treat everything else as UTF-8 text.
    return raw.decode("utf-8", errors="replace")


def _ok(fn, *args):
    try:
        return fn(*args)
    except ValueError as exc:        
        raise HTTPException(400, str(exc))
    except Exception as exc:         
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")


# ----- pipeline endpoints --------------------------------------------------- #
@app.post("/api/load-text")
def api_load_text(body: LoadTextBody):
    return _ok(rag.load_text, body.text, body.source_name)


@app.post("/api/load-file")
def api_load_file(file: UploadFile = File(...)):
    text = _read_upload(file)
    return _ok(rag.load_text, text, file.filename or "uploaded-file")


@app.post("/api/chunk")
def api_chunk(body: ChunkBody):
    return _ok(rag.chunk_text, body.chunk_size, body.chunk_overlap)


@app.post("/api/embed")
def api_embed():
    return _ok(rag.embed_and_store)


@app.post("/api/retrieve")
def api_retrieve(body: RetrieveBody):
    return _ok(rag.retrieve, body.question, body.top_k)


@app.post("/api/generate")
def api_generate(body: GenerateBody):
    return _ok(rag.generate, body.question, body.top_k)


@app.post("/api/run-all")
def api_run_all(body: RunAllBody):
    return _ok(rag.run_all, body.question)


@app.get("/api/status")
def api_status():
    return {
        "has_text": bool(rag.state.raw_text),
        "source": rag.state.source_name,
        "chunk_count": len(rag.state.chunks),
        "embedded": rag.state.vector_store is not None,
        "settings": {
            "chunk_size": rag.state.chunk_size,
            "chunk_overlap": rag.state.chunk_overlap,
            "top_k": rag.state.top_k,
        },
    }


# ----- serve the frontend --------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
