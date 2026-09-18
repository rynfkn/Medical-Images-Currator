from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from app.api import annotations, auth, cases, datasets, files, reviews

app = FastAPI(title="Medical Dataset Curation", version="0.1.0")
for router in (
    auth.router,
    datasets.router,
    cases.router,
    annotations.router,
    reviews.router,
    files.router,
):
    app.include_router(router, prefix="/api/v1")


@app.exception_handler(IntegrityError)
async def integrity_error(request: Request, exc: IntegrityError):
    return JSONResponse(status_code=409, content={"detail": "Conflicting database record"})
