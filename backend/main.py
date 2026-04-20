"""
main.py — Entry point del backend FastAPI.

Avvio in sviluppo:
    uvicorn main:app --reload --port 8000

Docs interattive (auto-generate):
    http://localhost:8000/docs
"""

from dotenv import load_dotenv
load_dotenv()  # DEVE essere prima degli import di core.*

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import pipeline, nli, interpret, dataset

# ── Istanza app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Post-Generation Citation System — API",
    description="Backend REST per il sistema di citazione post-hoc.",
    version="0.2.0",
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Registra i router ───────────────────────────────────────────────────────
app.include_router(pipeline.router)
app.include_router(nli.router)
app.include_router(interpret.router)
app.include_router(dataset.router)


# ── Endpoint base ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "citation-backend"}


@app.get("/")
async def root():
    return {
        "message": "Citation Backend API",
        "docs": "/docs",
        "health": "/health",
    }