import shutil
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import Admin, CurrentUser
from app.db import Db
from app.models import AnnotationFormat, Dataset, ImageFormat
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


def staged_name(upload: UploadFile, suffixes: tuple[str, ...]) -> str:
    """Client file names are never trusted as paths, only as leaf names."""
    name = Path(upload.filename or "").name
    if not name or name.startswith(".") or name in ("..", "."):
        raise HTTPException(422, "Every uploaded file needs a name")
    if not name.lower().endswith(suffixes):
        raise HTTPException(422, f"{name} is not one of {', '.join(suffixes)}")
    return name


def stage(upload: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    limit = get_settings().max_upload_bytes
    written = 0
    with target.open("xb") as stream:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                raise HTTPException(413, f"{target.name} exceeds the upload size limit")
            stream.write(chunk)


@router.post("/{dataset_id}/upload", status_code=201)
def upload_dataset_files(
    dataset_id: UUID,
    db: Db,
    user: Admin,
    images: Annotated[list[UploadFile], File()],
    labels: Annotated[list[UploadFile], File()] = [],  # noqa: B006 - FastAPI form default
    annotations: Annotated[UploadFile | None, File()] = None,
):
    """Ingest files uploaded from the browser, staged under IMPORT_DIR and then removed."""
    dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id).with_for_update())
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    if not images:
        raise HTTPException(422, "Select at least one file")
    root = get_settings().import_dir.resolve() / "uploads" / str(dataset_id) / uuid4().hex
    try:
        if dataset.image_format == ImageFormat.DICOM:
            for upload in images:
                stage(upload, root / staged_name(upload, (".dcm",)))
            handler = ingest_dicom_dataset
        elif dataset.image_format == ImageFormat.NIFTI:
            for upload in images:
                stage(upload, root / "images" / staged_name(upload, (".nii", ".nii.gz")))
            for upload in labels:
                stage(upload, root / "labels" / staged_name(upload, (".nii", ".nii.gz")))
            handler = ingest_nifti_dataset
        else:
            suffix = (".png",) if dataset.image_format == ImageFormat.PNG else (".jpg", ".jpeg")
            for upload in images:
                stage(upload, root / "images" / staged_name(upload, suffix))
            if dataset.annotation_format == AnnotationFormat.COCO:
                if annotations is None:
                    raise HTTPException(422, "A COCO dataset needs an instances.json file")
                stage(annotations, root / "annotations" / "instances.json")
            handler = ingest_coco_dataset
        try:
            count = handler(db, dataset, import_path(str(root)), user)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
    finally:
        # Ingestion copied what it needs into DATA_DIR; the staged copy is disposable.
        shutil.rmtree(root, ignore_errors=True)
    return {"dataset_id": dataset.id, "cases_ingested": count}
