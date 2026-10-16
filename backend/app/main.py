from fastapi import FastAPI

from app.routers import auth, chapters, jobs, relevance, show_notes, transcripts, uploads

app = FastAPI(title="YouTube Suite API")

app.include_router(auth.router)
app.include_router(uploads.router)
app.include_router(jobs.router)
app.include_router(transcripts.router)
app.include_router(chapters.router)
app.include_router(show_notes.router)
app.include_router(relevance.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
