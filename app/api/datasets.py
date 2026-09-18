from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.core.security import Admin, CurrentUser
from app.db import Db
from app.models import Dataset, ImageFormat
from app.schemas import DatasetCreate, DatasetOut, IngestRequest
from app.services.ingestion import ingest_coco_dataset, ingest_dicom_dataset, ingest_nifti_dataset
from app.services.storage import import_path

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(body: DatasetCreate, db: Db, user: Admin):
    dataset = Dataset(**body.model_dump())
    db.add(dataset)
    db.flush()
    return dataset


@router.get("", response_model=list[DatasetOut])
def list_datasets(
    db: Db, user: CurrentUser, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)
):
    return db.scalars(
        select(Dataset).order_by(Dataset.created_at, Dataset.id).offset(offset).limit(limit)
    ).all()


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: UUID, db: Db, user: CurrentUser):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset


@router.post("/{dataset_id}/ingest", status_code=201)
def ingest_dataset(dataset_id: UUID, body: IngestRequest, db: Db, user: Admin):
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id).with_for_update())
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    handlers = {
        "NIFTI": ({ImageFormat.NIFTI}, ingest_nifti_dataset),
        "COCO": ({ImageFormat.PNG, ImageFormat.JPEG}, ingest_coco_dataset),
        "DICOM": ({ImageFormat.DICOM}, ingest_dicom_dataset),
    }
    handler = handlers.get(body.type)
    if not handler or dataset.image_format not in handler[0]:
        raise HTTPException(422, "Ingestion type does not match dataset format")
    root = import_path(body.path)
    try:
        count = handler[1](db, dataset, root, user)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"dataset_id": dataset.id, "cases_ingested": count}
