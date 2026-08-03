from fastapi import FastAPI

app = FastAPI(title="Travel Assistant API", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "Travel Assistant API"}


@app.get("/health")
async def health():
    return {"status": "ok"}
