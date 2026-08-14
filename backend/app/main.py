from fastapi import FastAPI

from app.routers import auth, jobs, uploads

app = FastAPI(title="YouTube Suite API")

app.include_router(auth.router)
app.include_router(uploads.router)
app.include_router(jobs.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
