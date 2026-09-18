from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from app.api import annotations, auth, cases, datasets, files, reviews
from app.core.config import get_settings

app = FastAPI(title="Medical Dataset Curation", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,  # Authentication uses explicit bearer headers, not cookies.
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Range"],
    expose_headers=["Content-Disposition", "Content-Length", "Content-Range", "Accept-Ranges"],
)
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
