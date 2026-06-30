# RAG Lab — learn LangChain & RAG from scratch

A small, honest project for understanding Retrieval-Augmented Generation. Instead of hiding the pipeline behind one magic function, this lab breaks RAG into its six real steps and shows you the actual output of each one in the browser: the chunks, the vectors, the similarity scores, the exact prompt, and the answer.

You learn by watching data flow through the pipeline and by turning the knobs.

```
  01 Load → 02 Chunk → 03 Embed → 04 Store → 05 Retrieve → 06 Generate
```

## The one idea behind RAG

A plain language model answers from memory. It cannot know your private documents or anything after its training cutoff, and it will sometimes invent answers. RAG fixes this by **retrieving relevant passages from your own data at question time and putting them in front of the model**, so the answer is grounded in facts you supplied.

Everything in this project exists to make that single idea visible.

## The six steps (and what to look for)

| # | Step | What happens | What to inspect in the UI |
|---|------|--------------|---------------------------|
| 01 | **Load** | A file or pasted text becomes plain text | character/word counts, raw preview |
| 02 | **Chunk** | Text is split into small overlapping pieces | how many chunks, where the splits land, the overlap |
| 03 | **Embed** | Each chunk becomes a vector (a list of numbers) | vector dimensions, the first numbers of a real vector |
| 04 | **Store** | Vectors go into a searchable database (Chroma) | the vectors are now queryable |
| 05 | **Retrieve** | Your question is embedded and matched to the nearest chunks | which chunks matched, and their similarity scores |
| 06 | **Generate** | The chunks + question go to an LLM for a grounded answer | the **exact prompt** sent, and the answer |

The most important thing to internalise: **steps 1–5 never call a language model.** Retrieval is pure similarity search over your data. The LLM only appears at step 6. Most of RAG's quality is won or lost before the model is ever involved.

## Setup

You need **Python 3.10+** (3.12 recommended).

```bash
# 1. from the project folder, create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies (first run also downloads a small embedding model)
pip install -r requirements.txt

# 3. start the server
uvicorn main:app --reload --app-dir backend

# 4. open the lab
#    http://localhost:8000
```

Click **"Load the sample document"**, type a question such as *"Who invented the lens used in lighthouses?"*, and press **Run whole pipeline**. Then click each stage to inspect it.

### About API keys (you mostly don't need one)

Embedding and retrieval run **locally and free** using a small open model, so you can learn the heart of RAG with no account and no cost. A key is only needed for the final answer (step 6).

- **No key?** The lab runs in *dry-run* mode and shows you the exact prompt it would have sent. The whole retrieval pipeline still works.
- **Want real answers?** Copy `.env.example` to `.env` and fill in one provider:
  - **Anthropic** (Claude) — set `ANTHROPIC_API_KEY`
  - **OpenAI** — set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`
  - **Ollama** (fully local, no key) — install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`, set `LLM_PROVIDER=ollama`

## A learning path — do these in order

Reading about RAG teaches you the words; changing the knobs teaches you the behaviour. Run each experiment and watch the inspector.

1. **See chunking.** Load the sample. Set chunk size to 1500, run, look at step 02 — a few big chunks. Now set it to 150 and re-run — many tiny chunks. Big chunks blur meaning; tiny chunks fragment it.
2. **See overlap.** Set overlap to 0, then to 200, and compare consecutive chunks at step 02. Overlap stops sentences from being orphaned at a boundary.
3. **See retrieval.** Ask a question whose answer is in one specific paragraph. At step 05, check the similarity scores. A good match scores high; an off-topic question scores low across the board.
4. **See top-k matter.** Set top-k to 1 and ask a question that needs two facts from different paragraphs — the answer often misses one. Raise top-k to 6 and the model gets enough context.
5. **See grounding.** Ask something *not* in the document. A well-behaved RAG system should say it doesn't know, because the prompt (step 06) instructs the model to answer only from the retrieved context.
6. **See the prompt.** This is the payoff. At step 06, read the exact text sent to the model. RAG is, in the end, just *good search that builds a good prompt.*

## Where the concepts live in the code

Read these two files — they are the project.

- `backend/rag_pipeline.py` — the six steps, one function each, heavily commented. **Start here.**
- `backend/config.py` — provider selection and the no-key fallback.
- `backend/main.py` — the web server; each step is its own endpoint so the UI can advance one stage at a time.
- `frontend/` — the glass-box UI (plain HTML/CSS/JS, no build step).

## Ideas to extend it (next steps)

Once the basics click, try adding:

- **Better retrieval:** a re-ranker over the top results, or *multi-query* retrieval that rephrases the question several ways to improve recall.
- **Real loaders:** LangChain has loaders for web pages, Word, Notion and more — swap the loader in step 1.
- **Persistence:** point Chroma at a folder so the index survives restarts.
- **Evaluation:** measure whether answers are faithful to the retrieved context (look up the RAGAS framework).
- **Streaming + memory:** stream tokens to the browser and keep conversation history.
- **Agentic RAG:** let the model decide *when* to search rather than always retrieving (LangGraph). Add this only when a task genuinely needs multi-step planning — start simple.

## Notes

- A retrieved passage can itself contain text that looks like an instruction ("ignore previous instructions…"). This is *prompt injection*. The system prompt in `rag_pipeline.py` tells the model to treat retrieved context as data only. No mitigation is perfect; it is an inherent limitation of putting data and instructions in the same context window.
- Embedding model: `all-MiniLM-L6-v2` (small, fast, 384 dimensions). For higher quality at the cost of speed, switch `embedding_model` in `config.py` to `all-mpnet-base-v2`.
